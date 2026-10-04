"""Заготовки задач на NumPy."""

import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices = data.matrices
    vectors = data.vectors
    n = len(matrices)
    result = np.zeros_like(vectors[0], dtype=float)
    for i in range(n):
        prod = matrices[i] @ vectors[i]
        result = result + prod
    return result

def binarize(data: BinarizeInput) -> np.ndarray:
    matrix = data.matrix
    threshold = data.threshold
    new_matrix = np.zeros_like(matrix, dtype=int)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] > threshold:
                new_matrix[i, j] = 1
            else:
                new_matrix[i, j] = 0
    return new_matrix

def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    result = []
    for i in range(matrix.shape[0]):
        row = matrix[i]
        unique = np.unique(row).tolist()
        result.append(unique)
    return result

def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    result = []
    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        unique = np.unique(col).tolist()
        result.append(unique)
    return result

def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows = data.rows
    columns = data.columns
    mean = data.mean
    std = data.std
    seed = data.seed
    
    if seed is not None:
        np.random.seed(seed)
    
    matrix = np.random.normal(mean, std, (rows, columns))
    
    row_means = matrix.mean(axis=1)
    col_means = matrix.mean(axis=0)
    row_vars = matrix.var(axis=1)
    col_vars = matrix.var(axis=0)
    
    return MatrixStatistics(
        matrix=matrix,
        row_means=row_means,
        col_means=col_means,
        row_vars=row_vars,
        col_vars=col_vars
    )

def chess(data: ChessInput) -> np.ndarray:
    rows = data.rows
    columns = data.columns
    first = data.first
    second = data.second
    
    result = np.full((rows, columns), first, dtype=int)
    for i in range(rows):
        for j in range(columns):
            if (i + j) % 2 == 1:
                result[i, j] = second
    return result

def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width = data.width
    height = data.height
    image_height = data.image_height
    image_width = data.image_width
    shape_color = data.shape_color
    background_color = data.background_color
    
    img = np.zeros((image_height, image_width, 3), dtype=int)
    for i in range(image_height):
        for j in range(image_width):
            img[i, j] = background_color
    
    cy = image_height // 2
    cx = image_width // 2
    y1 = max(0, cy - height // 2)
    y2 = min(image_height, cy + height // 2)
    x1 = max(0, cx - width // 2)
    x2 = min(image_width, cx + width // 2)
    
    for i in range(y1, y2):
        for j in range(x1, x2):
            img[i, j] = shape_color
    
    return img

def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x = data.semi_axis_x
    semi_axis_y = data.semi_axis_y
    image_height = data.image_height
    image_width = data.image_width
    shape_color = data.shape_color
    background_color = data.background_color
    
    img = np.zeros((image_height, image_width, 3), dtype=int)
    for i in range(image_height):
        for j in range(image_width):
            img[i, j] = background_color
    
    cy = image_height / 2
    cx = image_width / 2
    
    for i in range(image_height):
        for j in range(image_width):
            value = ((j - cx)**2 / semi_axis_x**2) + ((i - cy)**2 / semi_axis_y**2)
            if value <= 1:
                img[i, j] = shape_color
    
    return img

def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values = data.values
    window = data.window
    
    n = len(values)
    
    # среднее
    total = 0
    for v in values:
        total = total + v
    mean = total / n
    
    # дисперсия
    total_var = 0
    for v in values:
        total_var = total_var + (v - mean) ** 2
    variance = total_var / n
    
    std = variance ** 0.5
    
    local_maxima = []
    local_minima = []
    for i in range(1, n - 1):
        if values[i] > values[i-1] and values[i] > values[i+1]:
            local_maxima.append(i)
        elif values[i] < values[i-1] and values[i] < values[i+1]:
            local_minima.append(i)
    
    moving_average = []
    for i in range(n - window + 1):
        window_sum = 0
        for j in range(window):
            window_sum = window_sum + values[i + j]
        moving_average.append(window_sum / window)
    moving_average = np.array(moving_average)
    
    return TimeSeriesStatistics(
        mean=mean,
        variance=variance,
        std=std,
        local_maxima=local_maxima,
        local_minima=local_minima,
        moving_average=moving_average
    )

def one_hot(data: OneHotInput) -> np.ndarray:
    labels = data.labels
    class_count = data.class_count
    
    if class_count is None:
        max_label = labels[0]
        for label in labels:
            if label > max_label:
                max_label = label
        class_count = max_label + 1
    
    result = np.zeros((len(labels), class_count), dtype=int)
    for i in range(len(labels)):
        result[i, labels[i]] = 1
    return result