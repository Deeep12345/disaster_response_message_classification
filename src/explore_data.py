def get_class_ratio(df, columns):
    """Get class ratio for columns in dataframe"""
    class_ratio = {}
    for col in columns:
        if df[col].nunique() == 1:
            continue
        class_counts = df[col].value_counts()
        n0 = class_counts.loc[0]
        n1 = class_counts.loc[1]
        if n0 < n1:
            smaller, larger = n0, n1
        else:
            smaller, larger = n1, n0
        class_ratio[col] = round(larger / smaller, 2)
    return class_ratio
