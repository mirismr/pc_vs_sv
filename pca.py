import numpy as np
from skspatial.objects import Plane
from skspatial.objects import Point
from skspatial.objects import Vector

def PCA(X, num_components):
    # center data in origin substracting mean
    X_meaned = X - np.mean(X, axis=0)

    # calculate covariance matrix
    cov_mat = np.cov(X_meaned, rowvar=False)

    # calculate autovalues and autovectors
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)

    # get the index ordered by autovalues
    sorted_index = np.argsort(eigen_values)[::-1]

    # sorted_eigenvalue = eigen_values[sorted_index] # we dont use these autovalues, only want the index sorted
    sorted_eigenvectors = eigen_vectors[:, sorted_index]

    # get the max n auto vectors
    eigenvector_subset = sorted_eigenvectors[:, 0:num_components]

    # calculate the new values
    X_reduced = np.dot(eigenvector_subset.transpose(), X_meaned.transpose()).transpose()

    return X_reduced, eigenvector_subset

def calculate_orthogonal_complement(x, normalize=True, threshold=1e-15):
    """Compute orthogonal complement of a matrix
    https://github.com/statsmodels/statsmodels/issues/3039

    this works along axis zero, i.e. rank == column rank,
    or number of rows > column rank
    otherwise orthogonal complement is empty
    """
    x = np.asarray(x)
    r, c = x.shape
    if r < c:
        import warnings
        warnings.warn('fewer rows than columns', UserWarning)

    # we assume svd is ordered by decreasing singular value, o.w. need sort
    s, v, d = np.linalg.svd(x)
    rank = (v > threshold).sum()

    oc = s[:, rank:]

    if normalize:
        k_oc = oc.shape[1]
        oc = oc.dot(np.linalg.inv(oc[:k_oc, :]))
    return oc


if __name__ == "__main__":
    matrix = np.array(
        [
            [1, 3],
            [2, 1],
            [3, 2],
            [-1, -2],
            [-2, -3],
            [-3, -1]
        ]
    )

    matrix2 = np.array(
        [
            [1, 5],
            [2, 4],
            [3, 3],
            [4, 2],
            [5, 1],
        ]
    )

    matrix3 = np.array(
        [
            [1, 3, -3],
            [2, 1, -1],
            [3, 2, -2],
            [-1, -2, 2],
            [-2, -3, 3],
            [-3, -1, 1]
        ]
    )
    sv_test = np.array(
        [
            [0.86],
            [0.47],
            [0.17]
        ]
    )

    _, suporting_vector = PCA(matrix3, 1)
    principal_components, suporting_vectors = PCA(matrix3, 2)
    orthogonal_complement = calculate_orthogonal_complement(suporting_vector)

    # la aplicacion recibe como entrada una matriz de puntos nx3 (probablemente R3)
    # como salida: suporting vector calculado desde "PCA"
    # primera componente principal y segunda componente principal de "PCA"
    # la segunda componente principal de PCA tiene que coincidir con la primera componente principal de los puntos proyectados (projected_points)

    print("****SUPORTING VECTORS****")
    print(suporting_vectors)
    print("****PRINCIPAL COMPONENTS****")
    print(principal_components)
    print("****ORTHOGONAL COMPLEMENT****")
    print(orthogonal_complement)

    v1 = Vector(orthogonal_complement.T[0, :])
    v2 = Vector(orthogonal_complement.T[1, :])
    point = Point([0, 0, 0]) # we suppose that the input matrix is centered at origin
    plane = Plane.from_vectors(point, v1, v2)

    projected_points = np.array([plane.project_point(x) for x in matrix3])
    print("****PROJECTED POINTS****")
    print(projected_points)

    first_principal_component, supporting_vector_projected_points = PCA(projected_points, 1)

    print("****FIRST PRINCIPAL COMPONENT****")
    print(first_principal_component)
    print("****SUPPORTING VECTOR PROJECTED POINTS****")
    print(supporting_vector_projected_points)
