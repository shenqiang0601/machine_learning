from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Python Spark CSV").getOrCreate()

# 利用inferSchema参数进行类型推断
df = spark.read.format('csv').option('header', 'true').option('inferSchema', 'true').load('GDM.csv')

df.show()

from pyspark.sql.functions import col

# 删除含有空值的行
df = df.dropna()

# 转换数据类型
df = df.withColumn("AGE", col("AGE").cast("integer"))

df.show()

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression

# 选择用于预测的特征列
assembler = VectorAssembler(
    inputCols=["GW", "APB", "APA1", "CRE", "CHOL", "UA","ALP", "TG", "GLU"],
    outputCol="features")

# 将数据集转换为适合机器学习的格式
output_data = assembler.transform(df)

# 构建逻辑回归模型
lr = LogisticRegression(featuresCol='features', labelCol='GDM')

# 划分数据集为训练集和测试集
train_data, test_data = output_data.randomSplit([0.7, 0.3])


# 训练模型
lr_model = lr.fit(train_data)

# 测试模型
predictions = lr_model.transform(test_data)

# 展示预测结果
#predictions.show()

from pyspark.ml.evaluation import BinaryClassificationEvaluator

# 构建评估器
evaluator = BinaryClassificationEvaluator()

predictions = predictions.withColumnRenamed("GDM", "label")
# 计算AUC
auc = evaluator.evaluate(predictions)

print(f"AUC: {auc}")