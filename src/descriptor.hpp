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
#include <numeric>
#include <vector>

namespace sycl_fft {

namespace detail {

// kernel names
template <typename Scalar, domain Domain, direction dir, detail::transpose transpose_in, int sg_size>
class buffer_kernel;
template <typename Scalar, domain Domain, direction dir, detail::transpose transpose_in, int sg_size>
class usm_kernel;
}  // namespace detail

// forward declaration
template <typename Scalar, domain Domain>
struct descriptor;

// specialization constants
constexpr static sycl::specialization_id<std::size_t> fft_size_spec_const{};

/*
Compute functions in the `committed_descriptor` call `dispatch_kernel` and `dispatch_kernel_helper`. These two functions
ensure the kernel is run with a supported subgroup size. Next `dispatch_kernel_helper` calls `run_kernel`. There are two
overloads of `run_kernel`, one for buffer interfaces and one for USM. `run_kernel` handles differences between forward
and backward computations, casts the memory (USM or buffers) from complex to scalars and launches the kernel.

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
  std::size_t n_compute_units;
  std::vector<std::size_t> supported_sg_sizes;
  sycl::kernel_bundle<sycl::bundle_state::executable> exec_bundle;
  int used_sg_size;
  Scalar* twiddles_forward;
  std::size_t local_memory_size;

  /**
   * Builds the kernel bundle with appropriate values of specialization constants for the first supported subgroup size.
   *
   * @tparam sg_size first subgroup size
   * @tparam other_sg_sizes other subgroup sizes
   * @return sycl::kernel_bundle<sycl::bundle_state::executable>
   */

  template <int sg_size, int... other_sg_sizes>
  sycl::kernel_bundle<sycl::bundle_state::executable> build_w_spec_const() {
    // This function is called from constructor initializer list and it accesses other data members of the class. These
    // are already initialized by the time this is called only if they are declared in the class definition before the
    // member that is initialized by this function.
    if (std::count(supported_sg_sizes.begin(), supported_sg_sizes.end(), sg_size)) {
      std::vector<sycl::kernel_id> ids;
      // if not used, some kernels might be optimized away in AOT compilation and not available here
      try {
        ids.push_back(sycl::get_kernel_id<detail::buffer_kernel<Scalar, Domain, direction::FORWARD,
                                                                detail::transpose::NOT_TRANSPOSED, sg_size>>());
      } catch (...) {
      }
      try {
        ids.push_back(sycl::get_kernel_id<detail::buffer_kernel<Scalar, Domain, direction::BACKWARD,
                                                                detail::transpose::NOT_TRANSPOSED, sg_size>>());
      } catch (...) {
      }
      try {
        ids.push_back(
            sycl::get_kernel_id<
                detail::usm_kernel<Scalar, Domain, direction::FORWARD, detail::transpose::NOT_TRANSPOSED, sg_size>>());
      } catch (...) {
      }
      try {
        ids.push_back(
            sycl::get_kernel_id<
                detail::usm_kernel<Scalar, Domain, direction::BACKWARD, detail::transpose::NOT_TRANSPOSED, sg_size>>());
      } catch (...) {
      }
      try {
        ids.push_back(
            sycl::get_kernel_id<
                detail::buffer_kernel<Scalar, Domain, direction::FORWARD, detail::transpose::TRANSPOSED, sg_size>>());
      } catch (...) {
      }
      try {
        ids.push_back(
            sycl::get_kernel_id<
                detail::buffer_kernel<Scalar, Domain, direction::BACKWARD, detail::transpose::TRANSPOSED, sg_size>>());
      } catch (...) {
      }
      try {
        ids.push_back(
            sycl::get_kernel_id<
                detail::usm_kernel<Scalar, Domain, direction::FORWARD, detail::transpose::TRANSPOSED, sg_size>>());
      } catch (...) {
      }
      try {
        ids.push_back(
            sycl::get_kernel_id<
                detail::usm_kernel<Scalar, Domain, direction::BACKWARD, detail::transpose::TRANSPOSED, sg_size>>());
      } catch (...) {
      }
      if (sycl::is_compatible(ids, dev)) {
        auto in_bundle = sycl::get_kernel_bundle<sycl::bundle_state::input>(queue.get_context(), ids);
        in_bundle.template set_specialization_constant<fft_size_spec_const>(params.lengths[0]);
        try {
          used_sg_size = sg_size;
          return sycl::build(in_bundle);
        } catch (std::exception& e) {
          std::cerr << "Build for subgroup size " << sg_size << " failed with message:\n" << e.what() << std::endl;
        }
      }
    }
    if constexpr (sizeof...(other_sg_sizes) == 0) {
      throw std::runtime_error("None of the compiled subgroup sizes are supported by the device!");
    } else {
      return build_w_spec_const<other_sg_sizes...>();
    }
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
        // get some properties we will use for tunning
        n_compute_units(dev.get_info<sycl::info::device::max_compute_units>()),
        supported_sg_sizes(dev.get_info<sycl::info::device::sub_group_sizes>()),
        // compile the kernels
        exec_bundle(build_w_spec_const<SYCLFFT_SUBGROUP_SIZES>()) {
    // TODO: check and support all the parameter values
    if (params.lengths.size() != 1) {
      throw std::runtime_error("SYCL-FFT only supports 1D FFT for now");
    }

    // get some properties we will use for tuning
    n_compute_units = dev.get_info<sycl::info::device::max_compute_units>();
    local_memory_size = queue.get_device().get_info<sycl::info::device::local_mem_size>();
    size_t factor1 = detail::factorize(params.lengths[0]);
    size_t factor2 = params.lengths[0] / factor1;
    // the local memory required for one fft and sub-fft twiddles
    std::size_t minimum_local_mem_required =
        (detail::num_scalars_in_local_mem<Scalar>(params.lengths[0], static_cast<std::size_t>(used_sg_size)) + factor1 +
         factor2) *
        sizeof(Scalar);
    if (!detail::fits_in_wi<Scalar>(factor2)) {
      if (minimum_local_mem_required > local_memory_size) {
        throw std::runtime_error("Insufficient amount of local memory available: " + std::to_string(local_memory_size) +
                                 " Required: " + std::to_string(minimum_local_mem_required));
      }
    }
    twiddles_forward = detail::calculate_twiddles<Scalar>(queue, params.lengths[0], used_sg_size);
    detail::populate_wg_twiddles<Scalar>(params.lengths[0], used_sg_size, twiddles_forward + 2 * (factor1 + factor2), queue);
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
    dispatch_kernel<direction::FORWARD>(in, out);
  }

  /**
   * Compute out of place backward FFT, working on buffers
   *
   * @param in buffer containing input data
   * @param out buffer containing output data
   */
  void compute_backward(const sycl::buffer<complex_type, 1>& in, sycl::buffer<complex_type, 1>& out) {
    dispatch_kernel<direction::BACKWARD>(in, out);
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
    return dispatch_kernel<direction::FORWARD>(in, out, dependencies);
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
    return dispatch_kernel<direction::BACKWARD>(in, out, dependencies);
  }

  committed_descriptor& operator=(const committed_descriptor&) = delete;
  committed_descriptor(const committed_descriptor&) = delete;

 private:
  /**
   * Dispatches the kernel with the first subgroup size that is supported by the device.
   *
   * @tparam dir FFT direction, takes either direction::FORWARD or direction::BACKWARD
   * @tparam T_in Type of the input buffer or USM pointer
   * @tparam T_out Type of the output buffer or USM pointer
   * @param in buffer or USM pointer to memory containing input data
   * @param out buffer or USM pointer to memory containing output data
   * @param dependencies events that must complete before the computation
   * @return sycl::event
   */
  template <direction dir, typename T_in, typename T_out>
  sycl::event dispatch_kernel(const T_in in, T_out out, const std::vector<sycl::event>& dependencies = {}) {
    return dispatch_kernel_helper<dir, T_in, T_out, SYCLFFT_SUBGROUP_SIZES>(in, out, dependencies);
  }

  /**
   * Helper for dispatching the kernel with the first subgroup size that is supported by the device.
   *
   * @tparam dir FFT direction, takes either direction::FORWARD or direction::BACKWARD
   * @tparam T_in Type of the input buffer or USM pointer
   * @tparam T_out Type of the output buffer or USM pointer
   * @tparam sg_size first subgroup size
   * @tparam other_sg_sizes other subgroup sizes
   * @param in buffer or USM pointer to memory containing input data
   * @param out buffer or USM pointer to memory containing output data
   * @param dependencies events that must complete before the computation
   * @return sycl::event
   */
  template <direction dir, typename T_in, typename T_out, int sg_size, int... other_sg_sizes>
  sycl::event dispatch_kernel_helper(const T_in in, T_out out, const std::vector<sycl::event>& dependencies = {}) {
    if (sg_size == used_sg_size) {
      std::size_t fft_size = params.lengths[0];  // 1d only for now
      std::size_t input_distance;
      std::size_t output_distance;
      Scalar scale_factor;
      if constexpr (dir == direction::FORWARD) {
        input_distance = params.forward_distance;
        output_distance = params.backward_distance;
        scale_factor = params.forward_scale;
      } else {
        input_distance = params.backward_distance;
        output_distance = params.forward_distance;
        scale_factor = params.backward_scale;
      }
      if (input_distance == fft_size && output_distance == fft_size) {
        return run_kernel<dir, detail::transpose::NOT_TRANSPOSED, sg_size>(in, out, scale_factor, dependencies);
      } else if (input_distance == 1 && output_distance == fft_size && in != out) {
        return run_kernel<dir, detail::transpose::TRANSPOSED, sg_size>(in, out, scale_factor, dependencies);
      } else {
        throw std::runtime_error("Unsupported configuration");
      }
    }
    if constexpr (sizeof...(other_sg_sizes) == 0) {
      throw std::runtime_error("None of the compiled subgroup sizes are supported by the device!");
    } else {
      return dispatch_kernel_helper<dir, T_in, T_out, other_sg_sizes...>(in, out, dependencies);
    }
  }

  /**
   * Common interface to run the kernel called by compute_forward and compute_backward
   *
   * @tparam dir FFT direction, takes either direction::FORWARD or direction::BACKWARD
   * @tparam transpose_in whether input is transposed (interpreting it as a matrix of batch size times FFT size)
   * @tparam Subgroup_size size of the subgroup
   * @tparam T_in Type of the input USM pointer
   * @tparam T_out Type of the output USM pointer
   * @param in USM pointer to memory containing input data
   * @param out USM pointer to memory containing output data
   * @param scale_factor Value with which the result of the FFT will be multiplied
   * @param dependencies events that must complete before the computation
   * @return sycl::event
   */
  template <direction dir, detail::transpose transpose_in, int Subgroup_size, typename T_in, typename T_out>
  sycl::event run_kernel(const T_in in, T_out out, Scalar scale_factor, const std::vector<sycl::event>& dependencies) {
    std::size_t n_transforms = params.number_of_transforms;
    std::size_t fft_size = params.lengths[0];  // 1d only for now
    std::size_t global_size = detail::get_global_size<Scalar>(fft_size, n_transforms, Subgroup_size, n_compute_units);
    auto in_scalar = reinterpret_cast<const Scalar*>(in);
    auto out_scalar = reinterpret_cast<Scalar*>(out);
    std::size_t twiddle_elements = detail::num_scalars_in_twiddles<Scalar>(fft_size, Subgroup_size);
    const Scalar* twiddles_ptr = twiddles_forward;
    std::size_t local_elements = detail::num_scalars_in_local_mem<Scalar>(fft_size, Subgroup_size);
    if constexpr (transpose_in == detail::transpose::TRANSPOSED) {
      local_elements =
          detail::num_scalars_in_local_mem<Scalar>(fft_size * SYCLFFT_SGS_IN_WG * Subgroup_size, Subgroup_size);
      if (local_elements * sizeof(Scalar) > local_memory_size) {
        throw std::runtime_error("Insufficient amount of local memory available: " + std::to_string(local_memory_size) +
                                 " Required: " + std::to_string(local_elements * sizeof(Scalar)));
      }
    }
    return queue.submit([&](sycl::handler& cgh) {
      cgh.depends_on(dependencies);
      cgh.use_kernel_bundle(exec_bundle);
      sycl::local_accessor<Scalar, 1> loc(local_elements, cgh);
      sycl::local_accessor<Scalar, 1> loc_twiddles(twiddle_elements, cgh);
      cgh.parallel_for<detail::usm_kernel<Scalar, Domain, dir, transpose_in, Subgroup_size>>(
          sycl::nd_range<1>{{global_size}, {Subgroup_size * SYCLFFT_SGS_IN_WG}},
          [=](sycl::nd_item<1> it, sycl::kernel_handler kh)
              [[sycl::reqd_sub_group_size(Subgroup_size)]] {
                detail::dispatcher<dir, transpose_in, Subgroup_size>(in_scalar, out_scalar, &loc[0], &loc_twiddles[0],
                                                      kh.get_specialization_constant<fft_size_spec_const>(),
                                                      n_transforms, it, twiddles_ptr, scale_factor);
              });
    });
  }

  /**
   * Common interface to dispatch compute called by compute_forward and compute_backward
   *
   * @tparam dir FFT direction, takes either direction::FORWARD or direction::BACKWARD
   * @tparam Subgroup_size size of the subgroup
   * @tparam transpose_in whether input is transposed (interpreting it as a matrix of batch size times FFT size)
   * @tparam T Type of buffer
   * @param in buffer containing input data
   * @param out buffer containing output data
   * @param scale_factor Value with which the result of the FFT will be multiplied
   * @param dependencies events that must complete before the computation
   */
  template <direction dir, detail::transpose transpose_in, int Subgroup_size, typename T>
  sycl::event run_kernel(const sycl::buffer<T, 1>& in, sycl::buffer<T, 1>& out, Scalar scale_factor,
                         const std::vector<sycl::event>& dependencies) {
    std::size_t n_transforms = params.number_of_transforms;
    std::size_t fft_size = params.lengths[0];  // 1d only for now
    std::size_t global_size = detail::get_global_size<Scalar>(fft_size, n_transforms, Subgroup_size, n_compute_units);
    auto in_scalar = in.template reinterpret<Scalar, 1>(2 * in.size());
    auto out_scalar = out.template reinterpret<Scalar, 1>(2 * out.size());
    std::size_t twiddle_elements = detail::num_scalars_in_twiddles<Scalar>(fft_size, Subgroup_size);
    const Scalar* twiddles_ptr = twiddles_forward;
    std::size_t local_elements = detail::num_scalars_in_local_mem<Scalar>(fft_size, Subgroup_size);
    if constexpr (transpose_in == detail::transpose::TRANSPOSED) {
      local_elements =
          detail::num_scalars_in_local_mem<Scalar>(fft_size * SYCLFFT_SGS_IN_WG * Subgroup_size, Subgroup_size);
      if (local_elements * sizeof(Scalar) > local_memory_size) {
        throw std::runtime_error("Insufficient amount of local memory available: " + std::to_string(local_memory_size) +
                                 " Required: " + std::to_string(local_elements * sizeof(Scalar)));
      }
    }
    return queue.submit([&](sycl::handler& cgh) {
      cgh.depends_on(dependencies);
      sycl::accessor in_acc{in_scalar, cgh, sycl::read_only};
      sycl::accessor out_acc{out_scalar, cgh, sycl::write_only};
      sycl::local_accessor<Scalar, 1> loc(local_elements, cgh);
      sycl::local_accessor<Scalar, 1> loc_twiddles(twiddle_elements, cgh);
      cgh.use_kernel_bundle(exec_bundle);
      cgh.parallel_for<detail::buffer_kernel<Scalar, Domain, dir, transpose_in, Subgroup_size>>(
          sycl::nd_range<1>{{global_size}, {Subgroup_size * SYCLFFT_SGS_IN_WG}},
          [=](sycl::nd_item<1> it, sycl::kernel_handler kh)
              [[sycl::reqd_sub_group_size(Subgroup_size)]] {
                detail::dispatcher<dir, transpose_in, Subgroup_size>(&in_acc[0], &out_acc[0], &loc[0], &loc_twiddles[0],
                                                      kh.get_specialization_constant<fft_size_spec_const>(),
                                                      n_transforms, it, twiddles_ptr, scale_factor);
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
