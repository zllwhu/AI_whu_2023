from fcnn_raw import fcnn_raw
from fcnn_sklearn import fcnn_sklearn
from knn_raw import knn_raw
from knn_sklearn import knn_sklearn

if __name__ == '__main__':
    print(f'----------------KNN SKLEARN----------------')
    knn_sklearn()

    print(f'------------------KNN RAW------------------')
    knn_raw()

    print(f'----------------FCNN SKLEARN---------------')
    fcnn_sklearn()

    print(f'------------------FCNN RAW-----------------')
    fcnn_raw()
