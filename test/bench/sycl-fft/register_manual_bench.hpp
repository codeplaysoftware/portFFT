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

#ifndef SYCL_FFT_BENCH_REGISTER_MANUAL_BENCH_HPP
#define SYCL_FFT_BENCH_REGISTER_MANUAL_BENCH_HPP

#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

#include "launch_bench.hpp"

static constexpr std::pair<std::string_view, std::string_view> ARG_KEYS[] = {
    {"domain", "d"},    {"lengths", "n"},   {"batch", "b"},  {"fwd_strides", "fs"}, {"bwd_strides", "bs"},
    {"fwd_dist", "fd"}, {"bwd_dist", "bd"}, {"scale", "sx"}, {"storage", "s"},      {"placement", "p"},
};

// Order must match with the ARG_KEYS array
enum key_idx {
  DOMAIN,
  LENGTHS,
  BATCH,
  FWD_STRIDES,
  BWD_STRIDES,
  FWD_DIST,
  BWD_DIST,
  SCALE,
  STORAGE,
  PLACEMENT,
};

using arg_map_t = std::unordered_map<std::string_view, std::string_view>;

class bench_error : public std::runtime_error {
 private:
  template <typename... Ts>
  std::string concat(const Ts&... args) {
    std::stringstream ss;
    (ss << ... << args);
    return ss.str();
  }

 public:
  template <typename... Ts>
  explicit bench_error(const Ts&... args) : std::runtime_error{concat(args...)} {}
};

class invalid_value : public bench_error {
 public:
  explicit invalid_value(const std::string_view& key, const std::string_view& value)
      : bench_error{"Invalid '", key, "' value: '", value, "'"} {}
};

arg_map_t get_arg_map(std::string_view arg) {
  arg_map_t arg_map;
  const char delimiter = ',';
  std::size_t delim_idx = arg.find(delimiter);
  std::string_view token = arg.substr(0, delim_idx);
  while (!token.empty()) {
    auto split_idx = token.find('=');
    if (split_idx == std::string_view::npos) {
      throw bench_error{"Invalid token '", token, "'"};
    }
    std::string_view key = token.substr(0, split_idx);
    std::string_view value = token.substr(split_idx + 1);
    if (arg_map.find(key) != arg_map.end()) {
      throw bench_error{"Key can only be specified once: '", key, "'"};
    }
    bool is_key_valid = std::any_of(std::begin(ARG_KEYS), std::end(ARG_KEYS),
                                    [&key](const std::pair<std::string_view, std::string_view>& pair) {
                                      return key == pair.first || key == pair.second;
                                    });
    if (!is_key_valid) {
      throw bench_error{"Invalid key: '", key, "'"};
    }
    if (value.empty()) {
      throw invalid_value(key, value);
    }
    arg_map[key] = value;

    // Last iteration sets delim_idx to npos
    if (delim_idx == std::string_view::npos) {
      break;
    }
    arg.remove_prefix(delim_idx + 1);
    delim_idx = arg.find(delimiter);
    token = arg.substr(0, delim_idx);
  }
  return arg_map;
}

std::string_view get_arg(arg_map_t& arg_map, key_idx key_idx) {
  const auto& key_names = ARG_KEYS[key_idx];
  auto long_it = arg_map.find(key_names.first);
  if (long_it != arg_map.end()) {
    return long_it->second;
  }
  auto short_it = arg_map.find(key_names.second);
  if (short_it != arg_map.end()) {
    return short_it->second;
  }
  return "";
}

std::size_t get_unsigned(const std::string_view& key, const std::string_view& value) {
  try {
    long size = std::stol(std::string(value));
    if (size <= 0) {
      throw std::exception();
    }
    return static_cast<std::size_t>(size);
  } catch (...) {
    throw bench_error{"Invalid '", key, "' value: '", value, "' must be a positive integer"};
  }
  return 0;
}

std::vector<std::size_t> get_vec_unsigned(const std::string_view& key, std::string_view value) {
  std::vector<std::size_t> vec;
  const char delimiter = 'x';
  std::size_t delim_idx = value.find(delimiter);
  std::string_view token = value.substr(0, delim_idx);
  while (!token.empty()) {
    vec.push_back(get_unsigned(key, token));

    // Last iteration sets delim_idx to npos
    if (delim_idx == std::string_view::npos) {
      break;
    }
    value.remove_prefix(delim_idx + 1);
    delim_idx = value.find(delimiter);
    token = value.substr(0, delim_idx);
  }
  return vec;
}

template <typename ftype, sycl_fft::domain domain>
void fill_descriptor(arg_map_t& arg_map, sycl_fft::descriptor<ftype, domain>& desc) {
  std::string_view arg = get_arg(arg_map, BATCH);
  if (!arg.empty()) {
    desc.number_of_transforms = get_unsigned("batch", arg);
  }

  arg = get_arg(arg_map, FWD_STRIDES);
  if (!arg.empty()) {
    desc.forward_strides = get_vec_unsigned("fwd_strides", arg);
  }

  arg = get_arg(arg_map, BWD_STRIDES);
  if (!arg.empty()) {
    desc.backward_strides = get_vec_unsigned("bwd_strides", arg);
  }

  arg = get_arg(arg_map, FWD_DIST);
  if (!arg.empty()) {
    desc.forward_distance = get_unsigned("fwd_dist", arg);
  }

  arg = get_arg(arg_map, BWD_DIST);
  if (!arg.empty()) {
    desc.backward_distance = get_unsigned("bwd_dist", arg);
  }

  arg = get_arg(arg_map, SCALE);
  if (!arg.empty()) {
    auto scale = static_cast<ftype>(std::stod(std::string(arg)));
    desc.forward_scale = scale;
    desc.backward_scale = scale;
  }

  arg = get_arg(arg_map, STORAGE);
  if (arg == "complex" || arg == "cpx") {
    desc.complex_storage = sycl_fft::complex_storage::COMPLEX;
  } else if (arg == "real_real" || arg == "rr") {
    desc.complex_storage = sycl_fft::complex_storage::REAL_REAL;
  } else if (!arg.empty()) {
    throw invalid_value{"storage", arg};
  }

  arg = get_arg(arg_map, PLACEMENT);
  if (arg == "in_place" || arg == "ip") {
    desc.placement = sycl_fft::placement::IN_PLACE;
  } else if (arg == "out_of_place" || arg == "oop") {
    desc.placement = sycl_fft::placement::OUT_OF_PLACE;
  } else if (!arg.empty()) {
    throw invalid_value{"placement", arg};
  }
}

template <typename ftype>
void register_benchmark(const std::string_view& desc_str) {
  using namespace sycl_fft;
  arg_map_t arg_map = get_arg_map(desc_str);

  // Set the domain and lengths first to create the descriptor
  domain domain;
  std::string_view domain_str = get_arg(arg_map, DOMAIN);
  if (domain_str == "complex" || domain_str == "cpx") {
    domain = domain::COMPLEX;
  } else if (domain_str == "real" || domain_str == "re") {
    domain = domain::REAL;
  } else if (domain_str.empty()) {
    throw bench_error{"'domain' must be specified"};
  } else {
    throw invalid_value{"domain", domain_str};
  }

  std::string_view lengths_str = get_arg(arg_map, LENGTHS);
  std::vector<std::size_t> lengths = get_vec_unsigned("lengths", lengths_str);
  if (lengths.empty()) {
    throw bench_error{"'lengths' must be specified"};
  }

  std::string_view ftype_str = typeid(ftype).name();
  std::stringstream real_bench_name;
  std::stringstream device_bench_name;
  real_bench_name << "real_time," << ftype_str << ":" << desc_str;
  device_bench_name << "device_time," << ftype_str << ":" << desc_str;
  if (domain == domain::COMPLEX) {
    descriptor<ftype, domain::COMPLEX> desc{lengths};
    fill_descriptor(arg_map, desc);
    benchmark::RegisterBenchmark(real_bench_name.str().c_str(), bench_dft_real_time<ftype, domain::COMPLEX>, desc)
        ->UseManualTime();
    benchmark::RegisterBenchmark(device_bench_name.str().c_str(), bench_dft_device_time<ftype, domain::COMPLEX>, desc)
        ->UseManualTime();
  } else if (domain == domain::REAL) {
    descriptor<ftype, domain::REAL> desc{lengths};
    fill_descriptor(arg_map, desc);
    benchmark::RegisterBenchmark(real_bench_name.str().c_str(), bench_dft_real_time<ftype, domain::REAL>, desc)
        ->UseManualTime();
    benchmark::RegisterBenchmark(device_bench_name.str().c_str(), bench_dft_device_time<ftype, domain::REAL>, desc)
        ->UseManualTime();
  } else {
    throw bench_error{"Unexpected domain: ", static_cast<int>(domain)};
  }
}

void print_help(const std::string_view& name) {
  auto print_keys = [](key_idx key) {
    std::stringstream ss;
    ss << "\t'" << ARG_KEYS[key].first << "', '" << ARG_KEYS[key].second << "'";
    return ss.str();
  };
  const int w = 25;
  auto cout_aligned = [w]() -> std::ostream& { return std::cout << std::left << std::setw(w); };
  // clang-format off
  std::cout << "Usage " << name << " [option]... [configuration]...\n";
  std::cout << "\nOptions:\n";
  std::cout << "\t--help, -h: Print this help message.\n";
  std::cout << "\nConfigurations:\n";
  std::cout << "Configurations are of the format <key>=<value>[,<key>=<value>]...\n";
  cout_aligned() << "\t<keys>" << "<values>\n";
  cout_aligned() << print_keys(DOMAIN)      << "Domain used.\n";
  cout_aligned() << "\t"                    << "  'complex', 'cpx' Use the complex domain.\n";
  cout_aligned() << "\t"                    << "  'real', 're' Use the real domain.\n";
  cout_aligned() << print_keys(LENGTHS)     << "N0[xNi]... where each Ni is a positive integer.\n";
  cout_aligned() << print_keys(BATCH)       << "N a positive integer, batch size of the problem.\n";
  cout_aligned() << "\t"                    << "  Default to 1.\n";
  cout_aligned() << print_keys(FWD_STRIDES) << "N0[xNi]... where each Ni is a positive integer.\n";
  cout_aligned() << "\t"                    << "  Ni is the stride used to access the (i+1) dimension of the input (or output) for the forward (resp. backward) direction.\n";
  cout_aligned() << "\t"                    << "  The size is in number of elements.\n";
  cout_aligned() << "\t"                    << "  Default value is set to assume continuous row-major input.\n";
  cout_aligned() << print_keys(BWD_STRIDES) << "N0[xNi]... where each Ni is a positive integer.\n";
  cout_aligned() << "\t"                    << "  Ni is the stride used to access the (i+1) dimension of the output (or input) for the forward (resp. backward) direction.\n";
  cout_aligned() << "\t"                    << "  The size is in number of elements.\n";
  cout_aligned() << "\t"                    << "  Default value is set to assume continuous row-major input.\n";
  cout_aligned() << print_keys(FWD_DIST)    << "N0[xNi]... where each Ni is a positive integer.\n";
  cout_aligned() << "\t"                    << "  Ni is the distance (i.e. batch stride) used to access the next batch of the input (or output) for the forward (resp. backward) direction.\n";
  cout_aligned() << "\t"                    << "  The size is in number of elements.\n";
  cout_aligned() << "\t"                    << "  Default value is set to the size of one transform.\n";
  cout_aligned() << print_keys(BWD_DIST)    << "N0[xNi]... where each Ni is a positive integer.\n";
  cout_aligned() << "\t"                    << "  Ni is the distance (i.e. batch stride) used to access the next batch of the output (or input) for the forward (resp. backward) direction.\n";
  cout_aligned() << "\t"                    << "  The size is in number of elements.\n";
  cout_aligned() << "\t"                    << "  Default value is set to the size of one transform.\n";
  cout_aligned() << print_keys(SCALE)       << "float number, scale used for both directions.\n";
  cout_aligned() << "\t"                    << "  Default to 1.\n";
  cout_aligned() << print_keys(STORAGE)     << "Storage used for complex domain.\n";
  cout_aligned() << "\t"                    << "  'complex', 'cpx' Output is stored in a single complex output container.\n";
  cout_aligned() << "\t"                    << "  'real', 're' Output is stored in 2 real output containers. The first container holds real parts and the second holds imaginary parts of the output.\n";
  cout_aligned() << "\t"                    << "  Default to 'complex'.\n";
  cout_aligned() << print_keys(PLACEMENT)   << "Placement used.\n";
  cout_aligned() << "\t"                    << "  'in_place', 'ip' Re-use the input container as the output one.\n";
  cout_aligned() << "\t"                    << "  'out_of_place', 'oop' Use separate input and output container.\n";
  cout_aligned() << "\t"                    << "  Default to 'out_of_place'.\n";
  std::cout << std::endl;
  // clang-format on
}

template <typename ftype>
void register_benchmarks(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    std::string_view arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      print_help(argv[0]);
      return;
    }
    register_benchmark<ftype>(arg);
  }
}

#endif  // SYCL_FFT_BENCH_REGISTER_MANUAL_BENCH_HPP
