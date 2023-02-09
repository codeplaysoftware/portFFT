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

#include <general/dispatcher.hpp>
#include <enums.hpp>

#include <sycl/sycl.hpp>

#include <vector>
#include <cstdint>
#include <complex>

namespace sycl_fft{

namespace detail{
// kernel names
template<typename Scalar, domain Domain>
class buffer_kernel;
template<typename Scalar, domain Domain>
class usm_kernel;
}

//forward declaration
template<typename Scalar, domain Domain>
struct descriptor;

/**
 * A commited descriptor that contains everything that is needed to run FFT.
 * 
 * @tparam Scalar type of the scalar used for computations
 * @tparam Domain domain of the FFT
 */
template<typename Scalar, domain Domain>
class commited_descriptor{
    using complex_type = std::complex<Scalar>;

    friend class descriptor<Scalar, Domain>;
    descriptor<Scalar, Domain> params;
    sycl::queue queue;
    sycl::device dev;
    sycl::context ctx;
    std::size_t n_compute_units;
    std::size_t buffer_kernel_subgroup_size;
    std::size_t usm_kernel_subgroup_size;

    commited_descriptor(const descriptor<Scalar, Domain>& params, sycl::queue& queue)
             : params{params}, queue(queue),dev(queue.get_device()),ctx(queue.get_context()){
        //TODO: check and support all the parameter values
        assert(params.lengths.size() == 1);
        assert(params.lengths[0] <= 64);

        // get some properties we will use for tunning
        n_compute_units = dev.get_info<sycl::info::device::max_compute_units>();
        //TODO: query for those
        buffer_kernel_subgroup_size = 32;
        usm_kernel_subgroup_size = 32;
    }

public:
    static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double>, "Scalar must be either float or double!");
    /**
     * Alias for `Scalar`.
     */
    using scalar_type = Scalar;
    /**
     * Alias for `Domain`.
     */
    static constexpr domain domain_value = Domain;

    /**
     * Computes in-place forward FFT, working on a buffer.
     * 
     * @param inout buffer containing input and output data
     */
    void compute_forward(sycl::buffer<complex_type, 1> &inout){
        // For now we can just call out-of-place implementation.
        // This might need to be changed once we implement support for large sizes that work in global memory.
        compute_forward(inout, inout);
    }

    /**
     * Computes out-of-place forward FFT, working on buffers.
     * 
     * @param in buffer containing input data
     * @param out buffer containing output data
     */
    void compute_forward(const sycl::buffer<complex_type, 1> &in, sycl::buffer<complex_type, 1> &out){
        std::size_t n_transforms = params.number_of_transforms;
        std::size_t fft_size = params.lengths[0]; //1d only for now
        std::size_t global_size = buffer_kernel_subgroup_size * 
                            std::min(4*n_compute_units, 
                                    (n_transforms + buffer_kernel_subgroup_size - 1) / buffer_kernel_subgroup_size);
        std::size_t input_distance = params.forward_distance;
        std::size_t output_distance = params.backward_distance;
        auto in_scalar = in.template reinterpret<Scalar,1>(2 * in.get_size());
        auto out_scalar = out.template reinterpret<Scalar,1>(2 * out.get_size());
        queue.submit([&](sycl::handler& cgh){
            auto in_acc = in_scalar.get_access<sycl::access::mode::read>(cgh);
            auto out_acc = out_scalar.get_access<sycl::access::mode::write>(cgh);
            sycl::local_accessor<Scalar,1> loc(2*fft_size*buffer_kernel_subgroup_size, cgh);
            cgh.parallel_for<detail::buffer_kernel<Scalar, Domain>>(sycl::nd_range<1>{{global_size}, 
                                                              {buffer_kernel_subgroup_size}},[=](sycl::nd_item<1> it){
                detail::workitem_dispatcher(in_acc, out_acc, loc, fft_size, n_transforms, input_distance, output_distance, it);
            });
        });
    }

    /**
    * Computes in-place forward FFT, working on USM memory.
    * 
    * @param inout USM pointer to memory containing input and output data
    * @param dependencies events that must complete before the computation
    * @return sycl::event associated with this computation
    */
    sycl::event compute_forward(complex_type* inout,
                                const std::vector<sycl::event>& dependencies = {}){
        // For now we can just call out-of-place implementation.
        // This might need to be changed once we implement support for large sizes that work in global memory.
        return compute_forward(inout, inout, dependencies);
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
                                const std::vector<sycl::event>& dependencies = {}){
        std::size_t n_transforms = params.number_of_transforms;
        std::size_t fft_size = params.lengths[0]; //1d only for now
        std::size_t global_size = usm_kernel_subgroup_size * 
                            std::min(4*n_compute_units, 
                                    (n_transforms + usm_kernel_subgroup_size - 1) / usm_kernel_subgroup_size);
        std::size_t input_distance = params.forward_distance;
        std::size_t output_distance = params.backward_distance;
        const Scalar* in_scalar = reinterpret_cast<const Scalar*>(in);
        Scalar* out_scalar = reinterpret_cast<Scalar*>(out);
        return queue.submit([&](sycl::handler& cgh){
            cgh.depends_on(dependencies);
            sycl::local_accessor<Scalar,1> loc(2*fft_size*usm_kernel_subgroup_size, cgh);
            cgh.parallel_for<detail::usm_kernel<Scalar, Domain>>(sycl::nd_range<1>{{global_size}, 
                                                           {usm_kernel_subgroup_size}},[=](sycl::nd_item<1> it){
                detail::workitem_dispatcher(in_scalar, out_scalar, loc, fft_size, n_transforms, input_distance, output_distance, it);
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
template<typename Scalar, domain Domain>
struct descriptor{
    std::vector<std::size_t> lengths;
    Scalar forward_scale = 1;
    Scalar backward_scale = 1;
    std::size_t number_of_transforms = 1;
    complex_storage complex_storage = complex_storage::COMPLEX;
    packed_format packed_format = packed_format::CONJUGATE_EVEN;
    placement placement = placement::OUT_OF_PLACE;
    std::vector<std::size_t> forward_strides;
    std::vector<std::size_t> backward_strides;
    std::size_t forward_distance = 1;
    std::size_t backward_distance = 1;
    //TODO: add TRANSPOSE, WORKSPACE and ORDERING if we determine they make sense
    
    /**
     * Construct a new descriptor object.
     * 
     * @param lengths size of the FFT transform
     */
    explicit descriptor(std::vector<std::size_t> lengths) : lengths(lengths), forward_strides{1}, 
                                                                                backward_strides{1}{
        //TODO: properly set default values for forward_strides, backward_strides, forward_distance, backward_distance
        forward_distance = lengths[0];
        backward_distance = lengths[0];
    }

    /**
     * Commits the descriptor, precalculating what can be done in advance.
     * 
     * @param queue queue to use for computations
     * @return commited_descriptor<Scalar, Domain> 
     */
    commited_descriptor<Scalar, Domain> commit(sycl::queue& queue){
        return {*this, queue};
    }
};

}

#endif
