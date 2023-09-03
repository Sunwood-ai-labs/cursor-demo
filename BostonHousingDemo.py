class BostonHousingDemo:
    # 初期化メソッド
    def __init__(self):
        # ASCIIアートを定義
        self.ascii_art = """
             ____||____
         ///////////\\
        ///////////  \\
        |    _     |  |
        |[] | | [] |[]|
        |   | |   _|  |
        """
        # ASCIIアートを出力
        print(self.ascii_art)

    # データをロードするメソッド
    def load_data(self):
        # データのURL
        data_url = "http://lib.stat.cmu.edu/datasets/boston"
        # pandasを使ってデータを読み込む
        raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
        # 入力データと目標データを作成
        X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
        y = raw_df.values[1::2, 2]
        # 入力データと目標データを返す
        return X, y

    # データを分割するメソッド
    def split_data(self, X, y):
        # データを訓練データとテストデータに分割
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # 分割したデータを返す
        return X_train, X_test, y_train, y_test

    # 特徴量を標準化するメソッド
    def standardize_features(self, X_train, X_test):
        # 標準化のためのインスタンスを作成
        scaler = StandardScaler()
        # 訓練データを標準化
        X_train = scaler.fit_transform(X_train)
        # テストデータを標準化
        X_test = scaler.transform(X_test)
        # 標準化したデータを返す
        return X_train, X_test

    # モデルをセットアップするメソッド
    def setup_model(self, X_train):
        # モデルのインスタンスを作成
        model = Sequential()
        # モデルに層を追加
        model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1))
        # モデルを返す
        return model

    # モデルをコンパイルするメソッド
    def compile_model(self, model):
        # モデルをコンパイル
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        # コンパイルしたモデルを返す
        return model

    # モデルを訓練するメソッド
    def train_model(self, model, X_train, y_train):
        # モデルを訓練
        model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)
        # 訓練したモデルを返す
        return model

    # モデルを評価するメソッド
    def evaluate_model(self, model, X_test, y_test):
        # モデルを評価
        loss, mae = model.evaluate(X_test, y_test, verbose=0)
        # 評価結果を出力
        print(f'テスト損失: {loss}')
        print(f'テストMAE: {mae}')
        # 評価結果を返す
        return loss, mae

    # 予測値を計算するメソッド
    def calculate_predicted_values(self, model, X_test):
        # 予測値を計算
        y_pred = model.predict(X_test)
        # 予測値を返す
        return y_pred

    # 実際の値と予測値の差を視覚化するメソッド
    def visualize_difference(self, y_test, y_pred):
        # プロットのサイズを設定
        plt.figure(figsize=(10, 6))
        # 実際の値をプロット
        plt.plot(range(y_test.shape[0]), y_test, color='blue', label='実際の値')
        # 予測値をプロット
        plt.plot(range(y_pred.shape[0]), y_pred, color='red', label='予測値')
        # タイトル、軸ラベル、凡例を設定
        plt.title('時間経過による実際の値と予測値の差')
        plt.xlabel('時間')
        plt.ylabel('価格')
        plt.legend()
        # プロットを表示
        plt.show()

demo = BostonHousingDemo()
X, y = demo.load_data()
X_train, X_test, y_train, y_test = demo.split_data(X, y)
X_train, X_test = demo.standardize_features(X_train, X_test)
model = demo.setup_model(X_train)
model = demo.compile_model(model)
model = demo.train_model(model, X_train, y_train)
loss, mae = demo.evaluate_model(model, X_test, y_test)
y_pred = demo.calculate_predicted_values(model, X_test)
demo.visualize_difference(y_test, y_pred)
