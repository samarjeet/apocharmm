// BEGINLICENSE
//
// This file is part of chcuda, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Antti-Pekka Hynninen, Samarjeet Prasad
//
// ENDLICENSE

#pragma once
#include <memory>

#include <iostream>
class object_t {
public:
  object_t(int i) : _num(i) {}
  object_t(object_t &&other) = default;

  // copy assignment
  object_t &operator=(const object_t &other) = default;
  ~object_t() = default;
  int getNum() const { return _num; }

private:
  int _num;
};

class TestForce {
public:
  TestForce(int i) : object(std::make_unique<object_t>(i)), num(i) {
    ptr_obj = new object_t(4);
    ptr_obj[0] = object_t(2);
    ptr_obj[1] = object_t(3);
    ptr_obj[2] = object_t(4);
    ptr_obj[3] = object_t(5);
  }

  TestForce(TestForce &&other) {
    object = std::move(other.object);
    ptr_obj = other.ptr_obj;
    num = other.num;

    other.ptr_obj = nullptr;
    other.object.reset();
    other.num = 0;
  }

  ~TestForce() { delete ptr_obj; }

  void calc_force();

private:
  std::unique_ptr<object_t> object;
  object_t *ptr_obj;
  int num;
};
