#pragma once

#include <iostream>
#include <Python.h>
#include <numpy/arrayobject.h>

namespace py_api
{

void init_python();
extern PyObject *module;
void init_py_util();

}

