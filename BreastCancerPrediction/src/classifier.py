# Imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import TomekLinks 

# Driver function for classifiers
def main():

    # EDA
    EDA()

    # KNN classifier
    tune_and_test_knn()

	# Classification ree classifier
    tune_and_test_tree_classifier()

#----- KNN -----#
def tune_and_test_knn():
    train_set, validation_set_data, validation_set_results, test_set_data, test_set_results = get_datasets_knn()
    k = tune_parameters_knn(train_set, validation_set_data, validation_set_results)
    print('optimal k:', k)
    predictions = knn_classify(train_set, test_set_data, k)
    evaluate(predictions, test_set_results)

def tune_parameters_knn(train_set, validation_set_data, validation_set_results):
    k = 1
    optimal_k = k
    optimal_accuracy = 0
    while k <= 25:
        predictions = knn_classify(train_set, validation_set_data, k) 
        temp_accuracy = evaluate(predictions, validation_set_results)
        if temp_accuracy > optimal_accuracy:
            optimal_k = k
            optimal_accuracy = temp_accuracy
        k += 2
    return optimal_k

def knn_classify(train_set, test_set, k):
    predictions = []
    for test_instance in test_set:
        predictions.append(knn_classifier(train_set, test_instance, k))
    return predictions

def knn_classifier(train_set, test_instance, k):
    distances = []
    for train_instance in train_set:
        distances.append((euclidean_distance(train_instance[1:], test_instance), train_instance[0]))
    distances.sort(key = lambda tuple: tuple[0])

    # Shephard's method
    M_count = 0
    B_count = 0
    for i in range(k):
        if distances[i][1] == 0:
            B_count += (1 / distances[i][0])
        else:
            M_count += (1 / distances[i][0])

    if B_count >= M_count:
        return 0
    else:
        return 1

def euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

#----- Classification tree -----#
def tune_and_test_tree_classifier():
    train_set, validation_set, test_set = get_datasets_tree()

    max_depth, min_samples_split, min_info_gain = tune_parameters_tree(train_set, validation_set)
    print('optimal combination: min_samples = ', min_samples_split, ' , min_info_gain = ', min_info_gain)

    classifier = DecisionTree(max_depth, min_samples_split, min_info_gain)
    X_train = train_set.iloc[:, 1:].values
    Y_train = train_set.iloc[:, 0].values.reshape(-1, 1)
    classifier.fit(X_train, Y_train)

    X_test = test_set.iloc[:, 1:].values
    Y_test = test_set.iloc[:, 0].values.reshape(-1, 1)
    Y_pred = classifier.predict(X_test) 
    evaluate(Y_test, Y_pred)

def tune_parameters_tree(train_set, validation_set):
    X_validation = validation_set.iloc[:, 1:].values
    Y_validation = validation_set.iloc[:, 0].values.reshape(-1, 1)
    X_train = train_set.iloc[:, 1:].values
    Y_train = train_set.iloc[:, 0].values.reshape(-1, 1)

    start_min_samples_split = 10
    min_info_gain = 0

    optimal_accuracy = 0
    while min_info_gain <= 0.25:
        min_samples_split = start_min_samples_split
        while min_samples_split <= 25:
            classifier = DecisionTree(min_samples_split, min_info_gain)
            classifier.fit(X_train, Y_train)
            Y_pred = classifier.predict(X_validation) 
            print('combination: min_samples = ', min_samples_split, ' , min_info_gain = ', min_info_gain)
            temp_accuracy = evaluate(Y_validation, Y_pred)
            if temp_accuracy >= optimal_accuracy:
                optimal_accuracy = temp_accuracy
                optimal_combination = [min_samples_split,  min_info_gain]
            min_samples_split += 5
        min_info_gain += 0.05
    return optimal_combination

class treeNode():
    def __init__(self, feature_index = None, threshold = None, left_child = None, right_child = None, score = None, target_class = None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.score = score
        self.target_class = target_class

class DecisionTree():
    def __init__(self, min_samples_split, min_info_gain):
        
        self.root = None
        self.min_samples_split = min_samples_split
        self.min_info_gain = min_info_gain
        
    def build_tree(self, dataset, curr_depth = 0):
        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)
        if num_samples >= self.min_samples_split:
            best_split = self.get_best_split(dataset, num_features)
            if len(best_split) != 0 and best_split["info_gain"] > self.min_info_gain:
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth + 1)
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth + 1)
                return treeNode(best_split["feature_index"], best_split["threshold"], 
                            left_subtree, right_subtree, best_split["info_gain"])
        leaf_value = self.calculate_leaf_value(Y)
        return treeNode(target_class = leaf_value)
    
    def get_best_split(self, dataset, num_features):
        best_split = {}
        max_info_gain = - float("inf")
        
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            for threshold in possible_thresholds:
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    curr_info_gain = self.information_gain(y, left_y, right_y)
                    if curr_info_gain > max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain
        return best_split
    
    def split(self, dataset, feature_index, threshold):
        dataset_left = np.array([row for row in dataset if row[feature_index] <= threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index] > threshold])
        return dataset_left, dataset_right
    
    def information_gain(self, parent_node, left_child, right_child):
        left_child_weight = len(left_child)/len(parent_node)
        right_child_weight = len(right_child)/len(parent_node)
        gain = self.entropy(parent_node) - (left_child_weight * self.entropy(left_child) + right_child_weight * self.entropy(right_child))
        return gain
    
    def entropy(self, y):
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy
    
    def calculate_leaf_value(self, Y):
        Y = list(Y)
        return max(Y, key = Y.count)
    
    def fit(self, X, Y):
        dataset = np.concatenate((X, Y), axis = 1)
        self.root = self.build_tree(dataset)
    
    def predict(self, X):
        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions
    
    def make_prediction(self, x, tree):
        if tree.target_class != None: 
            return tree.target_class
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.make_prediction(x, tree.left_child)
        else:
            return self.make_prediction(x, tree.right_child)

#----- Evaluation -----#
def evaluate(predictions, test_set_results):
    total_correct = 0
    for i in range(len(predictions)):
        if predictions[i] == test_set_results[i]:
            total_correct += 1
    accuracy = total_correct * 100/len(predictions)
    print('Accuracy: ' + str(total_correct) + '/' + str(len(predictions)) + ": ", total_correct * 100/len(predictions), "%")
    return accuracy

#----- Data management -----#
def get_datasets_knn():
    train_set, feature_medians = get_dataset_knn('train')
    test_set = get_dataset_knn('test')
    medians_dict = feature_medians.to_dict()
    test_set = impute_test_set(test_set, medians_dict)
    test_set_data = test_set.drop('diagnosis', inplace = False, axis = 1)
    test_set_results = test_set['diagnosis']

    # Convert data to numpy arrays
    train_set, validation_set = train_test_split(train_set, test_size= 0.30, random_state = 3)
    validation_set_data = validation_set.drop('diagnosis', inplace = False, axis = 1)
    validation_set_results = validation_set['diagnosis']
    train_set = train_set.to_numpy()
    test_set_data = test_set_data.to_numpy()
    test_set_results = test_set_results.to_numpy()
    validation_set_data = validation_set_data.to_numpy()
    validation_set_results = validation_set_results.to_numpy()
    return train_set, validation_set_data, validation_set_results, test_set_data, test_set_results

def get_datasets_tree():
    train_set, feature_medians = get_dataset_knn('train')
    test_set = get_dataset_knn('test')
    medians_dict = feature_medians.to_dict()
    test_set = impute_test_set(test_set, medians_dict)
    train_set, validation_set = train_test_split(train_set, test_size = 0.30, random_state = 1000)

    train_set = balance_dataset_tomek_links(train_set)
    return train_set, validation_set, test_set

def balance_dataset_tomek_links(train_set):
    column_names = train_set.columns
    Xy = train_set.to_numpy()
    X = Xy[:, 1:]
    y = Xy[:, 0]
    tl = TomekLinks(sampling_strategy='majority')
    X_res, y_res = tl.fit_resample(X, y)
    y_res = y_res.reshape((len(y_res), 1))
    combined = np.concatenate([y_res, X_res], axis = 1)
    train_set = pd.DataFrame(combined, columns = column_names)
    return train_set

def get_dataset_tree(data_type):
    if data_type == 'train':
        df = pd.read_csv('data/breastCancerTrain.csv', header = 0, sep='\t')
    elif data_type == 'test':
        df = pd.read_csv('data/breastCancerTest.csv', header = 0, sep='\t')

    # Remove unwanted columns
    df.rename(columns = {'Bratio':'NoneColumn', 'gender':'Bratio'}, inplace = True)
    df.drop([' ', 'id', 'NoneColumn'], inplace = True, axis = 1)

    # Map target classes to binary, B to 0, M to 1
    df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1})

    df = df.replace({'?': np.nan})
    df = df.astype(float)

    if data_type == 'train':
        df = df.apply(clip_outliers_fill_missing_values, axis = 0)
        feature_medians  = df.median()
        return df, feature_medians

    elif data_type == 'test':
        return df

def get_dataset_knn(data_type):
    if data_type == 'train':
        df = pd.read_csv('data/breastCancerTrain.csv', header = 0, sep='\t')
    elif data_type == 'test':
        df = pd.read_csv('data/breastCancerTest.csv', header = 0, sep='\t')

    # Remove unwanted columns
    df.rename(columns = {'Bratio':'NoneColumn', 'gender':'Bratio'}, inplace = True)
    df.drop([' ', 'id', 'NoneColumn'], inplace = True, axis = 1)

    # Map target classes to binary, B to 0, M to 1
    df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1})

    df = df.replace({'?': np.nan})
    df = df.astype(float)

    if data_type == 'train':
        df = df.apply(clip_outliers_fill_missing_values, axis = 0)
        print(df.describe())
        feature_medians  = df.median()
        normalised_df = (df - df.min())/(df.max() - df.min())
        return normalised_df, feature_medians

    elif data_type == 'test':
        return (df - df.min())/(df.max() - df.min())

def clip_outliers_fill_missing_values(feature):
    feature_median = feature.median()
    feature[feature < 0] = np.nan
    feature[feature > 90000] = np.nan
    feature.fillna(feature_median, inplace = True)
    return feature

def impute_test_set(test_set, medians_dict):
    return test_set.apply(lambda column: column.fillna(medians_dict[column.name]), axis = 0)

def EDA():
    df = pd.read_csv('data/breastCancerTrain.csv', header = 0, sep='\t')

    # Remove unwanted columns
    df.rename(columns = {'Bratio':'NoneColumn', 'gender':'Bratio'}, inplace = True)
    df.drop([' ', 'id', 'NoneColumn'], inplace = True, axis = 1)

    # Map target classes to binary, B to 0, M to 1
    df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1})

    df = df.replace({'?': np.nan})
    df = df.astype(float)

    print('Number of missing values for each feature: ', df.isna().sum())
    print('Number of rows with at least one missing value: ', df.isna().any(axis = 1).sum())
    print('Number of negative values: ', (df < 0).sum().sum())

    df1 = df.describe(include = 'all')

    df1.loc['dtype'] = df.dtypes
    df1.loc['size'] = len(df)
    df1.loc['% count'] = df.isnull().mean()

    print(df1)
    print(df.groupby('diagnosis').size())
    df = df.apply(clip_outliers_fill_missing_values, axis = 0)
    print(df.describe())

    feature_medians  = df.median()
    normalised_df = (df - df.min())/(df.max() - df.min())
    return normalised_df, feature_medians

if __name__ == '__main__':
    main()