"""
MiniTensor Release Bundle Generator
Creates the single-header distribution and release zip archive ready for GitHub Releases tab.
Zero external dependencies - uses Python standard library.
"""

import os
import sys
import zipfile
import shutil

VERSION = "1.1.0"
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RELEASE_DIR = os.path.join(REPO_ROOT, "release")
INCLUDE_DIR = os.path.join(REPO_ROOT, "include")

def generate_single_header():
    """Generates an amalgamated, standalone single-header file for GitHub Releases."""
    os.makedirs(RELEASE_DIR, exist_ok=True)
    single_header_path = os.path.join(RELEASE_DIR, "minitensor.hpp")

    # Read core tensor implementation
    tensor_hpp_path = os.path.join(INCLUDE_DIR, "minitensor", "tensor.hpp")
    with open(tensor_hpp_path, "r", encoding="utf-8") as f:
        tensor_content = f.read()

    # Read license
    license_path = os.path.join(REPO_ROOT, "LICENSE")
    license_text = ""
    if os.path.exists(license_path):
        with open(license_path, "r", encoding="utf-8") as f:
            license_text = f.read().strip()

    header_banner = f"""/*
 * ==============================================================================
 * MiniTensor v{VERSION} - Single-Header Standalone Release Distribution
 * A lightweight, dependency-free C++20 library for arbitrary-rank tensors.
 *
 * GitHub Repository: https://github.com/your-org/minitensor
 *
 * MIT License:
{license_text}
 * ==============================================================================
 */

#ifndef MINITENSOR_SINGLE_HEADER_HPP
#define MINITENSOR_SINGLE_HEADER_HPP

#define MINITENSOR_VERSION_MAJOR 1
#define MINITENSOR_VERSION_MINOR 1
#define MINITENSOR_VERSION_PATCH 0
#define MINITENSOR_VERSION_STRING "{VERSION}"

"""

    # Strip existing header guards from tensor.hpp
    cleaned_body = tensor_content
    cleaned_body = cleaned_body.replace("#ifndef MINITENSOR_TENSOR_HPP", "")
    cleaned_body = cleaned_body.replace("#define MINITENSOR_TENSOR_HPP", "")
    cleaned_body = cleaned_body.replace("#endif  // MINITENSOR_TENSOR_HPP", "")
    cleaned_body = cleaned_body.strip()

    extra_functions = """

namespace minitensor {

template <typename T>
[[nodiscard]] inline Tensor<T> matmul(const Tensor<T>& a, const Tensor<T>& b) {
    return a.matmul(b);
}

template <typename T>
[[nodiscard]] inline Tensor<T> matmul_blocked(const Tensor<T>& a,
                                             const Tensor<T>& b,
                                             std::size_t block_size = 64) {
    return a.matmul_blocked(b, block_size);
}

template <typename T>
[[nodiscard]] inline Tensor<T> matmul_parallel(const Tensor<T>& a,
                                              const Tensor<T>& b,
                                              std::size_t block_size = 64) {
    return a.matmul_parallel(b, block_size);
}

template <typename T>
[[nodiscard]] inline Tensor<T> tensordot(const Tensor<T>& a,
                                        const Tensor<T>& b,
                                        std::size_t axes = 2) {
    return a.tensordot(b, axes);
}

template <typename T>
[[nodiscard]] inline Tensor<T> tensordot(
    const Tensor<T>& a, const Tensor<T>& b,
    const std::vector<std::size_t>& axes_a,
    const std::vector<std::size_t>& axes_b) {
    return a.tensordot(b, axes_a, axes_b);
}

template <typename T>
[[nodiscard]] inline Tensor<T> transpose(const Tensor<T>& a) {
    return a.transpose();
}

template <typename T>
[[nodiscard]] inline Tensor<T> reshape(const Tensor<T>& a,
                                       const std::vector<std::size_t>& new_shape) {
    return a.reshape(new_shape);
}

}  // namespace minitensor
"""

    footer = "\n\n#endif  // MINITENSOR_SINGLE_HEADER_HPP\n"

    full_content = header_banner + cleaned_body + extra_functions + footer

    with open(single_header_path, "w", encoding="utf-8") as f:
        f.write(full_content)

    print(f"Generated single-header: {single_header_path} ({len(full_content)} bytes)")
    return single_header_path

def generate_release_zip(single_header_path):
    """Packages the release archive minitensor-v{VERSION}.zip for GitHub Releases."""
    zip_filename = os.path.join(RELEASE_DIR, f"minitensor-v{VERSION}.zip")
    
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add single-header
        zf.write(single_header_path, arcname=f"minitensor-v{VERSION}/minitensor.hpp")

        # Add include directory
        for root, _, files in os.walk(INCLUDE_DIR):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, REPO_ROOT)
                zf.write(abs_path, arcname=f"minitensor-v{VERSION}/{rel_path}")

        # Add root files
        for fname in ["LICENSE", "CMakeLists.txt", "README.md", "RELEASE_AND_USAGE.md"]:
            fpath = os.path.join(REPO_ROOT, fname)
            if os.path.exists(fpath):
                zf.write(fpath, arcname=f"minitensor-v{VERSION}/{fname}")

        # Add cmake config template
        cmake_in = os.path.join(REPO_ROOT, "cmake", "MiniTensorConfig.cmake.in")
        if os.path.exists(cmake_in):
            zf.write(cmake_in, arcname=f"minitensor-v{VERSION}/cmake/MiniTensorConfig.cmake.in")

    print(f"Generated release archive: {zip_filename} ({os.path.getsize(zip_filename)} bytes)")
    return zip_filename

if __name__ == "__main__":
    sh = generate_single_header()
    generate_release_zip(sh)
    print("Release bundle generated successfully in 'release/' directory.")
