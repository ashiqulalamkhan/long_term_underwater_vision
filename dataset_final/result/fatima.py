class DF:
    def __init__(self, name, sheet_name):

        self.name = name
        self.sheet_name = sheet_name
        self.df = self.compute_df()
        self.input_keys = self.get_input_keys()

    def remove_kPa(self, df):
        return [i for i in df.columns if not "kPa" in i]

    def compute_df(self):

        if "WAG" in self.name:
            df = pd.read_excel(self.name, self.sheet_name, index_col=0)
            df.columns = [i.strip() for i in df.columns]

            if self.sheet_name in ["Nominal", "Qstatic", "Dynamic"]:
                df["Material Grade"] = df["Cases"].str.split("/|\|").str[0]
                df = df.drop(columns=["Cases"])


            elif self.sheet_name == "Static":
                df["Material Grade"] = df["Material"]
                df = df.drop(columns=["Material"])


        elif "21565-RES-0006-03" in self.name:
            df = pd.read_excel(self.name, self.sheet_name, index_col=0)

            cols = list(df.iloc[13].values)
            df.columns = cols
            df = df.iloc[14:]

            cols = list(df.columns)
            cols[13] = "HOA (°)"
            df.columns = cols

            idx = df.columns.notna()
            df = df[df.columns[idx]]

        df.rename(columns={
            'Water Depth (m)': "WD (m)",
            "HOA(deg)": "HOA (°)",
            "DP(bar)": "DP (bar)",
            "Design Pressure (bar)": "DP (bar)",
            "Offset Direction": "Direction",
            "Nominal Hang-off angle (deg)": "HOA (°)",
            "Internal Fluid (kg/m3)": "Content Density",
            "Max DNVGL LRFD UR Along Entire Length (-)": 'DNVGL LRFD UR Along Entire Length (-)',
            "Min Effective Tension @Top (kN)": 'Effective Tension @Top (kN)',
        }, inplace=True)
        for i in df.columns:
            df[i] = df[i].replace(["None"], np.NaN)

        cols = self.remove_kPa(df)
        return df[cols]

    def get_input_keys(self):

        if self.sheet_name == "Nominal":
            input_keys = ['Material Grade', 'WD (m)', 'OD (in)', 'DP (bar)', 'Current', 'Offset (m)', 'Direction',
                          'Content Density', "HOA (°)"
                          ]

        elif self.sheet_name == "Static":
            input_keys = ["Material Grade", "WD (m)", "OD (in)", "DP (bar)", "Content Density", "Direction", "HOA (°)"]


        elif self.sheet_name == "Qstatic":
            input_keys = ["Material Grade", "WD (m)", "OD(in)", "DP (bar)", "Current", "Offset (m)", "Direction",
                          "Content Density", "HOA (°)"]


        elif self.sheet_name == "Dynamic":
            input_keys = ["Material Grade", "WD (m)", "OD (in)", "DP (bar)", "Current", "Offset (m)", "Direction",
                          "Content Density", "Harmonic Heave Motion Velocity (m/s)",
                          "Harmonic Heave Motion Amplitude (m)", "HOA (°)"]

            if "21565-RES-0006-03" in self.name:
                input_keys.append("Harmonic Heave Motion Period (s)")

        cols_without_nan = []
        for col in input_keys:
            if col in self.df.columns:
                if not self.df[col].isnull().values.any():
                    cols_without_nan.append(col)
                else:
                    print("remove col :", col)
                    self.df.drop(columns=[col], inplace=True)
        return cols_without_nan

    def get_output_keys(self):

        input_keys = set(self.get_input_keys())
        cols = set(self.df.columns)
        output_keys = list(cols.difference(input_keys))

        return output_keys

    def find_keys_to_iterate(self):

        dict_keys = {}

        input_keys = self.get_input_keys()

        for i in input_keys:
            cases = list(self.df[i].unique())
            cases.sort()
            if type(cases[0]) in [np.int64, np.float64]:
                if not any([math.isnan(x) for x in cases]):
                    dict_keys[i] = cases
            elif type(cases[0]) == str:
                dict_keys[i] = cases

        return dict_keys


def filter_with_ur_and_eff_tension(df):
    selector = df['DNVGL LRFD UR Along Entire Length (-)'] < 1
    selector &= df['Effective Tension @Top (kN)'] < 6000
    df_ = df[selector]
    return df_


def transform_to_is_x80(df):
    df = df.copy()
    df["is_x80"] = (df['Material Grade'] == "X80").astype(int)
    col_to_conserve = [col for col in df.columns if col != "Material Grade"]
    df = df[col_to_conserve]
    return df


##CONVERTING CLASS TO INTEGER VALUES
def get_categorical_cols(df):
    categorical_cols = []
    for col in df.columns:
        unique = list(df[col].unique())
        if type(unique[0]) == str:
            categorical_cols.append(col)
    return categorical_cols


def append_to_list(list_1, list_2):
    for i in list_2:
        list_1.append(i)
    return list_1


def dummies_categorical(df):
    cat_col = get_categorical_cols(df)
    all_new_cols = []
    cols_to_remove = []
    for col in cat_col:
        data = pd.get_dummies(df[col], prefix=col)

        all_new_cols = append_to_list(all_new_cols, data.columns)
        cols_to_remove.append(col)

        df = pd.concat([df, data], axis=1).drop([col], axis=1)
    return df, all_new_cols, cols_to_remove


def prepare_df(df, col_x):
    df_ = filter_with_ur_and_eff_tension(df)
    df_ = transform_to_is_x80(df_)
    df_, new_cols, cols_to_remove = dummies_categorical(df_)

    for col_to_remove in cols_to_remove:
        if col_to_remove in col_x:
            col_x.remove(col_to_remove)
    if len(new_cols) > 0:
        for new_col in new_cols:
            col_x.append(new_col)
    return df_, col_x


def remove_Current(col_x):
    print("Remove Current")
    return [i for i in col_x if "Current" not in i]


def compute_decision_tree(dataF, col_x, max_depth=4, tag="inputs"):
    df = dataF.df
    df_, col_x = prepare_df(df, col_x)
    col_x = remove_Current(col_x)

    col_y = "is_x80"

    clf = DecisionTreeClassifier(
        random_state=0,
        max_depth=max_depth

    )
    clf.fit(df_[col_x], df_[col_y])

    trees = tree.export_graphviz(
        clf,
        out_file=None,
        feature_names=col_x,
        class_names=["X65", "X80"],
        filled=True,
        rounded=True,
        special_characters=True,
        proportion=True
    )

    graph = pydotplus.graph_from_dot_data(trees)

    display(Image(graph.create_png()))

    if "WAG" in dataF.name:
        folder_1 = "WAG"
    elif "21565-RES-0006-03" in self.name:
        folder_1 = "H2"

    location = os.path.join("../plots", folder_1, "decision_tree")

    if not os.path.exists(location):
        os.makedirs(location)
    if tag == "inputs":
        filename = f"X65-X80_{folder_1}_{dataF.sheet_name}_inputs_decision_tree.png"
    elif tag == "inputs+2":
        filename = f"X65-X80_{folder_1}_{dataF.sheet_name}_inputs+2_decision_tree.png"
    elif tag == "inputs+3":
        filename = f"X65-X80_{folder_1}_{dataF.sheet_name}_inputs+3_decision_tree.png"

    graph.write_png(os.path.join(location, filename))