/*
Spherical Harmonics Calculations - calculating spherical harmonics in C++
Copyright (C) 2022 - Jack Naylor
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>. 
*/

#pragma once
#include <Eigen/Core>
#include <cmath>
#include <complex>

typedef unsigned int uint;

/**
 * @brief Binomial coefficients of the form: (n k) computed via dynamic programming
 * 
 * @param n 
 * @param k 
 * @return uint 
 */
uint binomial_coefficient(const uint& n, const uint& k) { // O(1) implementation for unsigned binomial coeff.

    uint answer = n - k + 1;

    for (int i = 1; i < k; i++) {
        answer *= (n - k + 1 + i) / (i + 1);
    }

    return answer;
}

/**
 * @brief Binomial coefficients of the form: (n k) computed via dynamic programming
 * 
 * @param n 
 * @param k 
 * @return int 
 */
int binomial_coefficient(const int& n, const int& k) { // O(1) implementation for int binomial coeff.
    assert(n >= 0 && k >= 0 );
    int answer = n - k + 1;

    for (int i = 1; i < k; i++) {
        answer *= (n - k + 1 + i) / (i + 1);
    }

    return answer;
}


/**
 * @brief Generalised binomial coefficient (non-int types). Uses std::tgamma
 * 
 * @param n 
 * @param k 
 * @return float
 */
auto binomial_coefficient(const auto& n, const auto& k) { // Generalised binomial coefficient... likely slow.
    return tgamma(n + 1)/(tgamma(k + 1) * tgamma(n - k + 1));
}


/**
 * @brief Factorial in pseudo-polynomial time.
 * 
 * @param x 
 * @return uint 
 */
uint factorial(const uint& x) { // Pseudo-polynomial implementation for unsigned factorial.
    uint answer = 1;

    for (int i = 1; i < x + 1; i++) {
        answer *= i;
    }

    return answer;
}

/**
 * @brief Factorial in pseudo-polynomial time.
 * 
 * @param x 
 * @return int 
 */
int factorial(const int& x) { // Pseudo-polynomial implementation for int factorial.
    
    assert(x >= 0);

    int answer = 1;

    for (int i = 1; i < x + 1; i++) {
        answer *= i;
    }

    return answer;
}


/**
 * @brief Compute Legendre polynomial value.
 * 
 * @param x Argument of function.
 * @param m Degree of polynomial.
 * @param l Order of polynomial.
 * @return auto 
 */
auto legendre_poly(const auto& x, const int& m, const int& l) { // Closed form of the Associated Legendre Polynomial.
    
    // If |m|>l, then P_ml(x) = 0
    // l<0, then P_m(-l)(x)=P_m(l-1)(x)
    // m<0, then perform proportional case
    // Base case.

    
    if (abs(m) > l) {
        return 0.0;
    } else if (l < 0) {
        return legendre_poly(x, m, abs(l) - 1);
    } else if (m < 0) {

        int pos_m = abs(m);

        return pow(-1,pos_m) * float(factorial(l - pos_m)) / factorial(l + pos_m) * legendre_poly(x, pos_m, l);
    } else {

        float sum = 0;

        for (int k = m; k < l + 1; k++) {
            sum += float(factorial(k)) / factorial(k-m) * pow(x, k-m) * binomial_coefficient(l, k) * binomial_coefficient(float(l + k - 1) / 2, l);
        }

        return pow(-1, m) * pow(2, l) * pow(1 - x * x, float(m) / 2) * sum;
    }
}

/**
 * @brief A_m function for Herglotzian form of spherical harmonics.
 * 
 * @param x 
 * @param y 
 * @param m degree
 * @return float 
 */

float a_func(const auto& x, const auto& y, const int& m) {
    float sum = 0;

    for (int p = 0; p < m + 1; p++) {
        sum += binomial_coefficient(m, p) * pow(x,p) * pow(y,m-p) * cos((m-p) * M_PI_2);
    }
    return sum;
}

/**
 * @brief B_m function for Herglotzian form of spherical harmonics.
 * 
 * @param x 
 * @param y 
 * @param m degree
 * @return float 
 */

float b_func(const auto& x, const auto& y, const int& m) {
    float sum = 0;

    for (int p = 0; p < m + 1; p++) {
        sum += binomial_coefficient(m, p) * pow(x, p) * pow(y, m - p) * sin((m - p) * M_PI_2);
    }
    return sum;
}


/**
 * @brief \f$\Pi_l^m$\f use in Herglotzian def of spherical harmonics.
 * 
 * @param z 
 * @param length 
 * @param m 
 * @param l 
 * @return float 
 */
float pi_func(const auto& z, const auto& length, const int& m, const int& l) {
    float sum = 0;

    for (int k = 0; k < (l - m) / 2 + 1; k++) {
        sum += pow(-1, k) * pow(2, -l) * binomial_coefficient(l, k) * binomial_coefficient(2 * l - 2 * k, l) * float(factorial(l - 2 * k)) / factorial(l - 2 * k - m) * pow(length, 2 * k) * pow(z, l - 2 * k - m);
    }

    return sqrt(float(factorial(l-m))/factorial(l+m))*sum;
}

bool in_2sphere(const auto& theta, const auto& phi) {
    return (abs(theta) <= M_PI_2 && abs(phi) <= M_PI);
}

/**
 * @brief Complex spherical harmonics (theta, phi).
 * 
 * @param theta 
 * @param phi 
 * @param m 
 * @param l 
 * @return std::complex<float> 
 */
std::complex<float> complex_spherical_harmonics(const auto& theta, const auto& phi, const int& m, const int& l) {
    assert(in_2sphere(theta, phi));

    return sqrt(float((2 * l + 1)) / (4 * M_PI) * float(factorial(l - m)) / factorial(l + m)) * legendre_poly(cos(theta), m, l) * std::exp(1 * std::complex_literals::i * m * phi);
}


/**
 * @brief Real spherical harmonics from (theta, phi).
 * 
 * @param theta 
 * @param phi 
 * @param m 
 * @param l 
 * @return auto 
 */
auto spherical_harmonics(const auto& theta, const auto& phi, const int& m, const int& l) {
    assert(in_2sphere(theta, phi));
    if (m < 0) {
        int pos_m = abs(m);
        return pow(-1, m) * sqrt(2) * sqrt(float((2 * l + 1)) / (4 * M_PI) * float(factorial(l - pos_m)) / factorial(l + pos_m)) * legendre_poly(cos(theta), pos_m, l) * sin(pos_m * phi);
    } else if (m == 0) {
        return sqrt((float(2 * l + 1)) / (4 * M_PI)) * legendre_poly(cos(theta), m, l);
    } else {
        return pow(-1, m) * sqrt(2) * sqrt(float((2 * l + 1)) / (4 * M_PI) * float(factorial(l - m)) / factorial(l + m)) * legendre_poly(cos(theta), m, l) * cos(m * phi);
    }
}

/**
 * @brief Complex spherical harmonics from Herglotzian definition.
 * 
 * @tparam T 
 * @param direction 
 * @param m 
 * @param l 
 * @return std::complex<float> 
 */
template <typename T>
std::complex<float> complex_spherical_harmonics(const Eigen::Vector3<T>& direction, const int& m, const int& l) {
    
    float dir_len = direction.norm();
    
    if (m > 0) {
        return pow(dir_len, -l) * sqrt(float(2 * l + 1) / (2 * M_PI)) * pi_func(direction[2], dir_len, m, l) * pow(-1, m) * (a_func(direction[0], direction[1], m) + 1 * std::complex_literals::i * b_func(direction[0], direction[1], m));
    } else if (m == 0) {
        return pow(dir_len, -l) * sqrt(float(2 * l + 1) / (2 * M_PI)) * pi_func(direction[2], dir_len, m, l);
    } else {
        return pow(dir_len, -l) * sqrt(float(2 * l + 1) / (2 * M_PI)) * pi_func(direction[2], dir_len, abs(m), l)*(a_func(direction[0], direction[1], abs(m)) - 1 * std::complex_literals::i * b_func(direction[0], direction[1], abs(m)));
    }
}

/**
 * @brief Real spherical harmonics from Herglotzian definition.
 * 
 * @tparam T 
 * @param direction 
 * @param m 
 * @param l 
 * @return auto 
 */
template <typename T>
auto spherical_harmonics(const Eigen::Vector3<T>& direction, const int& m, const int& l) {

    float dir_len = direction.norm();
    
    if (m > 0) {
        return pow(dir_len, -l) * sqrt(float(2 * l + 1) / (2 * M_PI)) * pi_func(direction[2], dir_len, m, l) * a_func(direction[0], direction[1], m);
    } else if (m == 0) {
        return pow(dir_len, -l) * sqrt(float(2 * l + 1) / (2 * M_PI)) * pi_func(direction[2], dir_len, m, l);
    } else {
        return pow(dir_len, -l) * sqrt(float(2 * l + 1) / (2 * M_PI)) * pi_func(direction[2], dir_len, abs(m), l) * b_func(direction[0], direction[1], abs(m));
    }

}