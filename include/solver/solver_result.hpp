#pragma once
// SolverResult<T> — non-throwing error type (C++20 polyfill for std::expected).
//
// Usage:
//   SolverResult<double> r = solver.advance_result();
//   if (!r.ok()) { std::cerr << r.error().message; return 1; }
//   double dt = r.value();

#include <string>
#include <variant>
#include <stdexcept>
#include <utility>

struct SolverError {
    std::string message;
    int         code = 0;   // 0 = generic
};

template<class T>
class SolverResult {
    std::variant<T, SolverError> v_;
public:
    static SolverResult ok(T val)           { return SolverResult(std::in_place_index<0>, std::move(val)); }
    static SolverResult err(SolverError e)  { return SolverResult(std::in_place_index<1>, std::move(e)); }

    bool               ok()    const noexcept { return v_.index() == 0; }
    const T&           value() const          { return std::get<T>(v_); }
    T&&                value()                { return std::get<T>(std::move(v_)); }
    const SolverError& error() const          { return std::get<SolverError>(v_); }
    explicit operator bool() const noexcept   { return ok(); }

    T value_or_throw() const {
        if (!ok()) throw std::runtime_error(error().message);
        return value();
    }

private:
    template<std::size_t I, class U>
    SolverResult(std::in_place_index_t<I> idx, U&& val)
        : v_(idx, std::forward<U>(val)) {}
};

template<>
class SolverResult<void> {
    std::variant<std::monostate, SolverError> v_;
public:
    static SolverResult make_ok()          { return SolverResult(std::in_place_index<0>, std::monostate{}); }
    static SolverResult err(SolverError e) { return SolverResult(std::in_place_index<1>, std::move(e)); }

    bool               ok()    const noexcept { return v_.index() == 0; }
    const SolverError& error() const          { return std::get<SolverError>(v_); }
    explicit operator bool() const noexcept   { return ok(); }

private:
    template<std::size_t I, class U>
    SolverResult(std::in_place_index_t<I> idx, U&& val)
        : v_(idx, std::forward<U>(val)) {}
};
