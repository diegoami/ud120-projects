outliers_cuts = [0, 2, 5, 10]
core_features = ['poi']
ana_features  = ['exercised_stock_options', 'salary', 'total_stock_value', 'from_poi_to_this_person' , 'bonus', 'long_term_incentive', 'total_payments'	, 'restricted_stock' ]
all_features  = core_features + ana_features
crux_feat = ['exercised_stock_options', 'salary', 'total_stock_value']