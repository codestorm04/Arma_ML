import numpy
import json
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.datasets import load_digits 
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_linnerud
from sklearn.datasets import load_boston
from sklearn.datasets import load_breast_cancer





def dump_obj(dataset, data_name):
    obj = {}
    obj["x"] = dataset.data if type(dataset.data) == list else dataset.data.tolist()
    obj["x_names"] = dataset.feature_names if type(dataset.feature_names) == list else dataset.feature_names.tolist()
    obj["y"] = dataset.target if type(dataset.target) == list else dataset.target.tolist()
    obj["desc"] = dataset.DESCR
    # regression task without target names  
    if hasattr(dataset, 'target_names'):
        obj["y_names"] = dataset.target_names if type(dataset.target_names) == list else dataset.target_names.tolist()
    with open(data_name + '.data', 'w') as f:
        json.dump(obj, f)


load_fns = [[load_iris, 'iris'], [load_boston, 'boston'], [load_linnerud, 'linnerud'], [load_diabetes, 'diabetes'], [load_wine, 'wine'], [load_breast_cancer, 'breast_cancer']]
            # [load_digits, 'digits']] for images data, remove 'feature_names' attribute
for fn in load_fns:
    dataset = fn[0]()
    dump_obj(dataset, fn[1])

# convert the generated *.data files into *.cc files manually in the ../../include/datasets/. 
