import pickle
import pandas as pd
#
# f = open('../learning_models/win_30/Logistic_OvR.pickle', 'rb')
# logistic = pickle.load(f)
# f.close()
#
# f = open('../learning_models/win_30/selected_features.pickle', 'rb')
# features = pickle.load(f)
# f.close()
#
# df_coeffs = pd.DataFrame(data=logistic.coef_, index=logistic.classes_, columns=features)
# df_coeffs.to_csv('../learning_models/logistic_coefficients.csv', index=True)
# # print logistic.intercept_
# # print logistic.n_iter_

# f = open('../learning_models/win_3k/DeepNN.pickle', 'rb')
# deepnn = pickle.load(f)
# f.close()
# print deepnn.classes_
# print deepnn.is_classifier
# print deepnn.is_initialized
# print deepnn.get_parameters()

# index = []
# # columns.append('Case/Window')
#
# df_logs = pd.DataFrame() # columns=columns
# fold_prefs = ['models_win_', 'models_win_pwc_']
# win_posts = ['30', '35', '40', '45']
#
#
# for fold_pref in fold_prefs:
#     for win_post in win_posts:
#         name = fold_pref + win_post
#         print name
#
#         f = open('../learning_models/'+ name +'/Logistic_OvR.pickle', 'rb')
#         clf = pickle.load(f)
#         f.close()
#
#         f = open('../learning_models/'+ name +'/selected_features.pickle', 'rb')
#         features = pickle.load(f)
#         f.close()
#
#         df_coeffs = pd.DataFrame(data=clf.coef_, index=clf.classes_, columns=features)
#         df_coeffs.to_csv('../learning_models/coefficients_' + name + '.csv', index=True)
#
#         if fold_pref + win_post == 'models_win_30':
#             index.extend(clf.classes_)
#             index.append('maximum number of iteration across all classes')
#             df_logs['classifier'] = index
#
#         col = []
#         col.extend(clf.intercept_)
#         col.append(clf.n_iter_[0])
#
#         df_logs[fold_pref + win_post] = col
#
# # baseline - models_baseline
# f = open('../learning_models/models_baseline/Logistic_OvR.pickle', 'rb')
# clf = pickle.load(f)
# f.close()
#
# f = open('../learning_models/models_baseline/selected_features.pickle', 'rb')
# features = pickle.load(f)
# f.close()
#
# df_coeffs = pd.DataFrame(data=clf.coef_, index=clf.classes_, columns=features)
# df_coeffs.to_csv('../learning_models/coefficients_baseline.csv', index=True)
#
# col = []
# col.extend(clf.intercept_)
# col.append(clf.n_iter_[0])
#
# df_logs['Baseline'] = col
#
# # baseline without pwc - models_pwc
# f = open('../learning_models/models_pwc/Logistic_OvR.pickle', 'rb')
# clf = pickle.load(f)
# f.close()
#
# f = open('../learning_models/models_pwc/selected_features.pickle', 'rb')
# features = pickle.load(f)
# f.close()
#
# df_coeffs = pd.DataFrame(data=clf.coef_, index=clf.classes_, columns=features)
# df_coeffs.to_csv('../learning_models/coefficients_baseline_without_pwc.csv', index=True)
#
# col = []
# col.extend(clf.intercept_)
# col.append(clf.n_iter_[0])
#
# df_logs['Baseline w/o PWC'] = col
#
# print df_logs.columns
# df_logs.columns = ['Class_Intercept', 'Case E 3k', 'Case E 3.5k', 'Case E 4k', 'Case E 4.5k', 'Case G 3k', 'Case G 3.5k', 'Case G 4k', 'Case G 4.5k', 'Baseline', 'Baseline without Pair Wise Calculations']
# df_logs.to_csv('../learning_models/logistic_parameters.csv', index=False)

f = open('../learning_models/deepNN_params.pickle', 'rb') # classification_models
weights = pickle.load(f)
f.close()

print weights