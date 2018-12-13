#include "py_util.hpp"

using namespace std;

PyObject *py_util::module = nullptr;

void py_util::init_py_util()
{
    if (module == nullptr) {
        module = PyImport_ImportModule("utilities");
    }
}

void py_util::init_python()
{
    Py_Initialize();
    if (!Py_IsInitialized()) {
        cout << "Error: Failed to initialize" << endl;
        exit(1);
    }
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('./')");

    PyRun_SimpleString("sys.argv = []");
    PyRun_SimpleString("sys.argv.append('gomokuer')");

    init_py_util();
}
