#pragma once


#include <cstring>
#include <vector>

template<typename T>
cli::array<T>^ pincpy_a_v(std::vector<T>& v) {
    cli::array<T>^ a(gcnew cli::array<T>(v.size()));
    pin_ptr<T> a_ptr = &a[0];
    std::memcpy(a_ptr, v.data(), v.size() * sizeof(T));
    return a;
}

template<typename T>
std::vector<T> pincpy_v_a(cli::array<T>^ a) {
    auto v{ std::vector<T>(a->Length) };
    pin_ptr<T> a_ptr = &a[0];
    std::memcpy(v.data(), a_ptr, a->Length * sizeof(T));
    return v;
}