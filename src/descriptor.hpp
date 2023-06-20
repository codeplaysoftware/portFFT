/***************************************************************************
 *
 *  Copyright (C) Codeplay Software Ltd.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  Codeplay's SYCL-FFT
 *
 **************************************************************************/

#ifndef SYCL_FFT_DESCRIPTOR_HPP
#define SYCL_FFT_DESCRIPTOR_HPP

#include <enums.hpp>
#include <general/dispatcher.hpp>

#include <sycl/sycl.hpp>

#include <complex>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <vector>

namespace sycl_fft {

namespace detail {

// kernel names
template <typename Scalar, domain Domain, direction dir>
class buffer_kernel;
template <typename Scalar, domain Domain, direction dir>
class usm_kernel;
}  // namespace detail

// forward declaration
template <typename Scalar, domain Domain>
struct descriptor;

// specialization constants
constexpr static sycl::specialization_id<std::size_t> fft_size_spec_const{};

/*
Compute functions in the `committed_descriptor` call `dispatch_compute`. There are two overloads of `dispatch_compute`,
one for buffer interfaces and one for USM. `dispatch_compute` handles differences between forward and backward
computations, casts the memory (USM or buffers) from complex to scalars and launches the kernel.

Many of the parameters for the kernel, such as number of workitems launched and the required size of local allocations
are determined by the helpers `num_scalars_in_local_mem` and `get_global_size` from `dispatcher.hpp`. The kernel calls
`dispatcher`. From here on, each function has only one templated overload that handles both directions of transforms and
buffer and USM memory.

`dispatcher` determines which implementation to use for the particular FFT size and calls one of
the dispatcher functions for the particular implementation: `workitem_dispatcher` or `subgroup_dispatcher`. In case of
subgroup, it also factors the FFT size into one factor that fits into individual workitem and one that can be done
across workitems in a subgroup. `dispatcher` and all other device functions make no assumptions on the size of a work
group or the number of workgroups in a kernel. These numbers can be tuned for each device. TODO: currently we always
test one subgroup per workgroup, so more may or may not actually work correctly.

Both dispatcher functions are there to make the size of the FFT that is handled by the individual workitems compile time
constant. `subgroup_dispatcher` also calls `cross_sg_dispatcher` that makes the cross-subgroup factor of FFT size
compile time constant. They do that by using a switch on the FFT size for one workitem, before calling `workitem_impl`
or `subgroup_impl` respectively. The `_impl` functions take the FFT size for one workitem as a template parameter. Only
the calls that are determined to fit into available registers (depending on the value of SYCLFFT_TARGET_REGS_PER_WI
macro) are actually instantiated.

The `workitem_impl` and `subgroup_impl` functions iterate over the batch of problems, loading data for each first in
local memory then from there into private one. This is done in these two steps to avoid non-coalesced global memory
accesses. `workitem_impl` loads one problem per workitem and `subgroup_impl` loads one problem per subgroup. After doing
computations by the calls to `wi_dft` for workitem and `sg_dft` for subgroup the data is written out, going through
local memory again.

The computational parts of the implementations are further documented in files with their implementations `workitem.hpp`
and `subgroup.hpp`.
*/

/**
 * A committed descriptor that contains everything that is needed to run FFT.
 *
 * @tparam Scalar type of the scalar used for computations
 * @tparam Domain domain of the FFT
 */
template <typename Scalar, domain Domain>
class committed_descriptor {
  using complex_type = std::complex<Scalar>;

  friend struct descriptor<Scalar, Domain>;
  descriptor<Scalar, Domain> params;
  sycl::queue queue;
  sycl::device dev;
  sycl::context ctx;
  sycl::kernel_bundle<sycl::bundle_state::executable> exec_bundle;
  std::size_t n_compute_units;
  Scalar* twiddles_forward;

  /**
   * Builds the kernel bundle with appropriate values of specialization constants.
   *
   * @return sycl::kernel_bundle<sycl::bundle_state::executable>
   */
  sycl::kernel_bundle<sycl::bundle_state::executable> build_w_spec_const() {
    // This function is called from constructor initializer list and it accesses other data members of the class. These
    // are already initialized by the time this is called only if they are declared in the class definition before the
    // member that is initialized by this function.
    auto in_bundle = sycl::get_kernel_bundle<sycl::bundle_state::input>(queue.get_context());
    in_bundle.set_specialization_constant<fft_size_spec_const>(params.lengths[0]);
    return sycl::build(in_bundle);
  }

  /**
   * Constructor.
   *
   * @param params descriptor this is created from
   * @param queue queue to use when enqueueing device work
   */
  committed_descriptor(const descriptor<Scalar, Domain>& params, sycl::queue& queue)
      : params{params},
        queue(queue),
        dev(queue.get_device()),
        ctx(queue.get_context()),
        exec_bundle(build_w_spec_const()) {
    // TODO: check and support all the parameter values
    if (params.lengths.size() != 1) {
      throw std::runtime_error("SYCL-FFT only supports 1D FFT for now");
    }

    // get some properties we will use for tuning
    n_compute_units = dev.get_info<sycl::info::device::max_compute_units>();
    twiddles_forward = detail::calculate_twiddles<Scalar>(queue, params.lengths[0], SYCLFFT_TARGET_SUBGROUP_SIZE);
  }

 public:
  static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double>,
                "Scalar must be either float or double!");
  /**
   * Alias for `Scalar`.
   */
  using scalar_type = Scalar;
  /**
   * Alias for `Domain`.
   */
  static constexpr domain domain_value = Domain;

  /**
   * Destructor
   */
  ~committed_descriptor() {
    queue.wait();
    if (twiddles_forward != nullptr) {
      sycl::free(twiddles_forward, queue);
    }
  }

  /**
   * Computes in-place forward FFT, working on a buffer.
   *
   * @param inout buffer containing input and output data
   */
  void compute_forward(sycl::buffer<complex_type, 1>& inout) {
    // For now we can just call out-of-place implementation.
    // This might need to be changed once we implement support for large sizes that work in global memory.
    compute_forward(inout, inout);
  }

  /**
   * Computes in-place backward FFT, working on a buffer.
   *
   * @param inout buffer containing input and output data
   */
  void compute_backward(sycl::buffer<complex_type, 1>& inout) {
    // For now we can just call out-of-place implementation.
    // This might need to be changed once we implement support for large sizes that work in global memory.
    compute_backward(inout, inout);
  }

  /**
   * Computes out-of-place forward FFT, working on buffers.
   *
   * @param in buffer containing input data
   * @param out buffer containing output data
   */
  void compute_forward(const sycl::buffer<complex_type, 1>& in, sycl::buffer<complex_type, 1>& out) {
    dispatch_compute_impl<direction::FORWARD, 1, complex_type>(in, out, params.forward_scale);
  }

  /**
   * Compute out of place backward FFT, working on buffers
   *
   * @param in buffer containing input data
   * @param out buffer containing output data
   */
  void compute_backward(const sycl::buffer<complex_type, 1>& in, sycl::buffer<complex_type, 1>& out) {
    dispatch_compute_impl<direction::BACKWARD, 1, complex_type>(in, out, params.backward_scale);
  }

  /**
   * Computes in-place forward FFT, working on USM memory.
   *
   * @param inout USM pointer to memory containing input and output data
   * @param dependencies events that must complete before the computation
   * @return sycl::event associated with this computation
   */
  sycl::event compute_forward(complex_type* inout, const std::vector<sycl::event>& dependencies = {}) {
    // For now we can just call out-of-place implementation.
    // This might need to be changed once we implement support for large sizes that work in global memory.
    return compute_forward(inout, inout, dependencies);
  }

  /**
   * Computes in-place backward FFT, working on USM memory.
   *
   * @param inout USM pointer to memory containing input and output data
   * @param dependencies events that must complete before the computation
   * @return sycl::event associated with this computation
   */
  sycl::event compute_backward(complex_type* inout, const std::vector<sycl::event>& dependencies = {}) {
    return compute_backward(inout, inout, dependencies);
  }

  /**
   * Computes out-of-place forward FFT, working on USM memory.
   *
   * @param in USM pointer to memory containing input data
   * @param out USM pointer to memory containing output data
   * @param dependencies events that must complete before the computation
   * @return sycl::event associated with this computation
   */
  sycl::event compute_forward(const complex_type* in, complex_type* out,
                              const std::vector<sycl::event>& dependencies = {}) {
    return dispatch_compute_impl<direction::FORWARD>(in, out, params.forward_scale, dependencies);
  }

  /**
   * Computes out-of-place backward FFT, working on USM memory.
   *
   * @param in USM pointer to memory containing input data
   * @param out USM pointer to memory containing output data
   * @param dependencies events that must complete before the computation
   * @return sycl::event associated with this computation
   */
  sycl::event compute_backward(const complex_type* in, complex_type* out,
                               const std::vector<sycl::event>& dependencies = {}) {
    return dispatch_compute_impl<direction::BACKWARD>(in, out, params.backward_scale, dependencies);
  }

 private:
  /**
   * Common interface to dispatch compute called by compute_forward and compute_backward
   *
   * @tparam dir FFT direction, takes either direction::FORWARD or direction::BACKWARD
   * @tparam Tin Type of the input USM pointer
   * @tparam Tout Type of the output USM pointer
   * @param in USM pointer to memory containing input data
   * @param out USM pointer to memory containing output data
   * @param scale_factor Value with which the result of the FFT will be multiplied
   * @param dependencies events that must complete before the computation
   * @return sycl::event
   */
  template <direction dir, typename Tin, typename Tout>
  inline sycl::event dispatch_compute_impl(const Tin in, Tout out, Scalar scale_factor = 1.0f,
                                           const std::vector<sycl::event>& dependencies = {}) {
    std::size_t n_transforms = params.number_of_transforms;
    std::size_t fft_size = params.lengths[0];  // 1d only for now
    const std::size_t subgroup_size = SYCLFFT_TARGET_SUBGROUP_SIZE;
    std::size_t global_size =
        detail::get_global_size<Scalar>(fft_size, params.number_of_transforms, subgroup_size, n_compute_units);
    auto in_scalar = reinterpret_cast<const Scalar*>(in);
    auto out_scalar = reinterpret_cast<Scalar*>(out);
    const Scalar* twiddles_ptr = twiddles_forward;
    std::size_t local_elements = detail::num_scalars_in_local_mem<Scalar>(fft_size, subgroup_size);
    return queue.submit([&](sycl::handler& cgh) {
      cgh.depends_on(dependencies);
      cgh.use_kernel_bundle(exec_bundle);
      sycl::local_accessor<Scalar, 1> loc(local_elements, cgh);
      sycl::local_accessor<Scalar, 1> loc_twiddles(fft_size * 2, cgh);
      cgh.parallel_for<detail::usm_kernel<Scalar, Domain, dir>>(
          sycl::nd_range<1>{{global_size}, {subgroup_size * SYCLFFT_SGS_IN_WG}}, [=
      ](sycl::nd_item<1> it, sycl::kernel_handler kh) [[sycl::reqd_sub_group_size(SYCLFFT_TARGET_SUBGROUP_SIZE)]] {
            detail::dispatcher<dir>(in_scalar, out_scalar, &loc[0], &loc_twiddles[0],
                                    kh.get_specialization_constant<fft_size_spec_const>(), n_transforms, it,
                                    twiddles_ptr, scale_factor);
          });
    });
  }

  /**
   * Common interface to dispatch compute called by compute_forward and compute_backward
   *
   * @tparam dir FFT direction, takes either direction::FORWARD or direction::BACKWARD
   * @tparam dim Number of dimension of the buffer
   * @tparam Tin Type of the input buffer
   * @tparam Tout Type of the output buffer
   * @param in buffer containing input data
   * @param out buffer containing output data
   * @param scale_factor Value with which the result of the FFT will be multiplied
   * @param dependencies events that must complete before the computation
   */
  template <direction dir, int dim, typename Tin, typename Tout>
  void dispatch_compute_impl(const sycl::buffer<Tin, dim>& in, sycl::buffer<Tout, dim>& out, Scalar scale_factor = 1.0f,
                             const std::vector<sycl::event>& dependencies = {}) {
    std::size_t n_transforms = params.number_of_transforms;
    std::size_t fft_size = params.lengths[0];  // 1d only for now
    const std::size_t subgroup_size = SYCLFFT_TARGET_SUBGROUP_SIZE;
    std::size_t global_size =
        detail::get_global_size<Scalar>(fft_size, params.number_of_transforms, subgroup_size, n_compute_units);
    auto in_scalar = in.template reinterpret<Scalar, dim>(2 * in.size());
    auto out_scalar = out.template reinterpret<Scalar, dim>(2 * out.size());
    const Scalar* twiddles_ptr = twiddles_forward;
    std::size_t local_elements = detail::num_scalars_in_local_mem<Scalar>(fft_size, subgroup_size);
    queue.submit([&](sycl::handler& cgh) {
      cgh.depends_on(dependencies);
      sycl::accessor in_acc{in_scalar, cgh, sycl::read_only};
      sycl::accessor out_acc{out_scalar, cgh, sycl::write_only};
      sycl::local_accessor<Scalar, 1> loc(local_elements, cgh);
      sycl::local_accessor<Scalar, 1> loc_twiddles(fft_size * 2, cgh);
      cgh.use_kernel_bundle(exec_bundle);
      cgh.parallel_for<detail::buffer_kernel<Scalar, Domain, dir>>(
          sycl::nd_range<1>{{global_size}, {subgroup_size * SYCLFFT_SGS_IN_WG}}, [=
      ](sycl::nd_item<1> it, sycl::kernel_handler kh) [[sycl::reqd_sub_group_size(SYCLFFT_TARGET_SUBGROUP_SIZE)]] {
            detail::dispatcher<dir>(&in_acc[0], &out_acc[0], &loc[0], &loc_twiddles[0],
                                    kh.get_specialization_constant<fft_size_spec_const>(), n_transforms, it,
                                    twiddles_ptr, scale_factor);
          });
    });
  }
};

/**
 * A descriptor containing FFT problem parameters.
 *
 * @tparam Scalar type of the scalar used for computations
 * @tparam Domain domain of the FFT
 */
template <typename Scalar, domain Domain>
struct descriptor {
  std::vector<std::size_t> lengths;
  Scalar forward_scale = 1;
  Scalar backward_scale = 1;
  std::size_t number_of_transforms = 1;
  complex_storage complex_storage = complex_storage::COMPLEX;
  placement placement = placement::OUT_OF_PLACE;
  std::vector<std::size_t> forward_strides;
  std::vector<std::size_t> backward_strides;
  std::size_t forward_distance = 1;
  std::size_t backward_distance = 1;
  // TODO: add TRANSPOSE, WORKSPACE and ORDERING if we determine they make sense

  /**
   * Construct a new descriptor object.
   *
   * @param lengths size of the FFT transform
   */
  explicit descriptor(std::vector<std::size_t> lengths) : lengths(lengths), forward_strides{1}, backward_strides{1} {
    // TODO: properly set default values for forward_strides, backward_strides, forward_distance, backward_distance
    forward_distance = lengths[0];
    backward_distance = lengths[0];
    for (auto l : lengths) {
      backward_scale *= Scalar(1) / static_cast<Scalar>(l);
    }
  }

  /**
   * Commits the descriptor, precalculating what can be done in advance.
   *
   * @param queue queue to use for computations
   * @return committed_descriptor<Scalar, Domain>
   */
  committed_descriptor<Scalar, Domain> commit(sycl::queue& queue) { return {*this, queue}; }

  std::size_t get_total_length() const noexcept {
    return std::accumulate(lengths.begin(), lengths.end(), 1LU, std::multiplies<std::size_t>());
  }
};

}  // namespace sycl_fft

#endif
