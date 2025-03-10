![image](https://raw.githubusercontent.com/NGO-Algorithm-Audit/python-synthpop/b09d3fe93ac21406199810e39e2a844dc1faefd0/images/Header.png)

# python-synthpop

Python implementation of the R package [synthpop](https://cran.r-project.org/web/packages/synthpop/index.html).

```python-synthpop``` is an open-source library for synthetic data generation (SDG). The library includes robust implementations of Classification and Regression Trees (CART) and Gaussian Copula (GC) synthesizers, equipping users with an open-source python library to generate high-quality, privacy-preserving synthetic data.

Synthetic data is generated in six steps:

1. **Detect data types**: detect numerical, categorial and/or datetime data;
2. **Process missing data**: process missing data: remove or impute missing values;
3. **Preprocessing**: transforms data into numerical format;
4. **Synthesizer**: fit CART or GC;
5. **Postprocessing**: map synthetic data back to its original structure and domain;
6. **Evaluation metrics**: determine quality of synthetic data, e.g., similarity, utility and privacy metrics. 

☁️ [Web app](https://algorithmaudit.eu/technical-tools/sdg/#web-app) – a demo of synthetic data generation using `python-synthpop` in the browser using [WebAssembly](https://github.com/NGO-Algorithm-Audit/local-first-web-tool).

# Installation

#### Pip

```
pip install python-synthpop
```

#### Source

```
git clone https://github.com/NGO-Algorithm-Audit/python-synthpop.git
cd python-synthpop
pip install -r requirements.txt
python setup.py install
```

# Examples

#### Social Diagnosis 2011 dataset
We will use the Social Diagnosis 2011 dataset as an example, which is a comprehensive survey conducted in Poland. This dataset includes a wide range of variables related to the social and economic conditions of Polish households and individuals. It covers aspects such as income, employment, education, health, and overall quality of life. 

```
In [1]:  import pandas as pd

In [2]:  df = pd.read_csv('../datasets/socialdiagnosis/data/SocialDiagnosis2011.csv', sep=';')
         df.head()
Out[2]:
	sex	age	marital	income	ls	smoke
0	FEMALE	57	MARRIED	800.0	PLEASED	NO
1	MALE	20	SINGLE	350.0	MOSTLY SATISFIED	NO
2	FEMALE	18	SINGLE	NaN	PLEASED	NO
3	FEMALE	78	WIDOWED	900.0	MIXED	NO
4	FEMALE	54	MARRIED	1500.0	MOSTLY SATISFIED	YES

```

### python-synthpop

Using default parameters the six steps are applied on the Social Diagnosis example tot generate synthetic data. See also [link](./example_notebooks/00_readme.ipynb).

```
In [1]:     from synthpop import MissingDataHandler, DataProcessor, CARTMethod

In [2]:     # 1. Initiate metadata
            metadata = MissingDataHandler()

            # 1.1 Detect data types
            column_dtypes = metadata.get_column_dtypes(df)
            print("Column Data Types:", column_dtypes)

            Column Data Types: {'sex': 'categorical', 'age': 'numerical', 'marital': 'categorical', 'income': 'numerical', 'ls': 'categorical', 'smoke': 'categorical'}

In [3]:     # 2. Missing data
            print(df.isnull().sum())

            sex          0
            age          0
            marital      9
            income     683
            ls           8
            smoke       10
            dtype: int64

In [4]:     # 2.1 Detect type of missingness
            missingness_dict = metadata.detect_missingness(df)
            print("Detected missingness yype:", missingness_dict)

            Detected missingness type: {'marital': 'MAR', 'income': 'MAR', 'ls': 'MAR', 'smoke': 'MAR'}


In [5]:     # 2.2 Impute missing values
            df_imputed = metadata.apply_imputation(df, missingness_dict)

            print(df_imputed.isnull().sum())

            sex        0
            age        0
            marital    0
            income     0
            ls         0
            smoke      0
            dtype: int64


In [6]:     # 3. Instantiate the DataProcessor with column types
            processor = DataProcessor(column_dtypes)

            # 3.1 Preprocess the data: transforms raw data into a numerical format
            processed_data = processor.preprocess(df)
            print("Processed Data:")
            display(processed_data.head())

            Processed Data:
            sex	age	marital	income	ls	smoke
            0	0	0.503625	3	-0.480608	4	0
            1	1	-1.495187	4	-0.834521	3	0
            2	0	-1.603231	4	NaN	4	0
            3	0	1.638086	5	-0.401961	1	0
            4	0	0.341559	3	0.069923	3	1

In [7]:     # 4. Fit the CART method
            cart = CARTMethod(metadata, smoothing=True, proper=True, minibucket=5, random_state=42)
            cart.fit(processed_data)


```