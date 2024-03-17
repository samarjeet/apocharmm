#pragma once
namespace charmm {
namespace constants {
constexpr double kBoltz =
    1.987191E-03; // Gas constant in AKMA units :  N_a * k_b / kcal
constexpr double atmosp = 1.4584007E-05;
constexpr double patmos = 1.0 / atmosp;
constexpr double surfaceTensionFactor = 98.6923; // converts dyne/cm to atm*angs
} // namespace constants
} // namespace charmm