#include "eval.hpp"

using namespace std;


int eval_init_numpy()
{
    if (PyArray_API == NULL) {
        import_array();
    }
}

eval::PyEvaluator::PyEvaluator(char *weight)
{
    init_succeeded = false;

    auto py_module = PyImport_ImportModule("model");
    auto py_dict = PyModule_GetDict(py_module);
    auto py_class = PyDict_GetItemString(py_dict, "Model");
    py_network = PyObject_CallFunction(py_class, "i", 0); // is_train=False

    eval_init_numpy();

    PyObject_CallMethod(py_network, "load_weight", "s", weight);
    if (!py_module) {
        cout << "Error: Failed to find model.py" << endl;
    }
    else if (!py_dict) {
        cout << "Error: Failed to get module dict" << endl;
    }
    else if (!py_class) {
        cout << "Error: Failed to get class" << endl;
    }
    else if (!PyCallable_Check(py_class)) {
        cout << "Error: Not callable." << endl;
    }
    else if (!py_network) {
        cout << "Error: Failed to init network" << endl;
    }
    else {
        init_succeeded = true;
    }

    Py_XDECREF(py_module);
    Py_XDECREF(py_dict);
    Py_XDECREF(py_class);
}

eval::PyEvaluator::~PyEvaluator()
{
    Py_XDECREF(py_network);
}

std::vector<eval::Evaluation*> eval::PyEvaluator::evaluate(std::vector<Game*> games, std::vector<Color> povs)
{
    std::vector<Evaluation*> evals;
    if (games.size() == 0) {
        return evals;
    }
    evals.reserve(games.size());
    auto obsvs = new Observation[games.size()];
    for (int ig = 0; ig < games.size(); ig++) {
        games[ig]->get_observation(obsvs[ig], povs[ig]);
    }

    npy_intp dims[4] = {games.size(), 2, BOARD_SIZE, BOARD_SIZE};
    auto py_obsv = PyArray_SimpleNewFromData(4, dims, NPY_FLOAT, obsvs);
    auto py_ret = PyObject_CallMethod(py_network, "evaluate", "O", py_obsv);

    Policy *policy;
    float *value;

    auto py_policy = PyTuple_GetItem(py_ret, 0);
    auto py_value = PyTuple_GetItem(py_ret, 1);
    PyArrayObject *arr_policy = NULL;
    PyArrayObject *arr_value = NULL;
    PyArray_GetArrayParamsFromObject(py_policy, NULL, 1, NULL, NULL, NULL, &arr_policy, NULL);
    PyArray_GetArrayParamsFromObject(py_value, NULL, 1, NULL, NULL, NULL, &arr_value, NULL);

    policy = (Policy*)PyArray_DATA(arr_policy);
    value = (float*)PyArray_DATA(arr_value);

    for (int ig = 0; ig < games.size(); ig++) {
        auto eval = new Evaluation;
        for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; i++) {
            eval->policy[i] = (*policy)[i];
            eval->value = *value;
        }
        evals.emplace_back(eval);
        policy++;
        value++;
    }

    Py_XDECREF(py_obsv);
    Py_XDECREF(py_ret);
    Py_XDECREF(py_policy);
    Py_XDECREF(py_value);
    delete[] obsvs;
    return evals;
}
