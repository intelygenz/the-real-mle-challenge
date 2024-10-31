## Challenge 1 - Refactor DEV code

The refactorization was done taking into consideration three main objectives:

- Time optimization
- Increase code maintainability
- Make the code testable at different stages

### Time optimization

To increase the time efficiency of the code, I analyzed the different fragments of code in the notebooks.
They were four fragments that could potentially be optimized. Mainly apply pandas functions. For each function to be refactored,
I created a function to apply the original and the refactored code to different data inputs, measure the time and created a plot.

Three of the refactored code snippets achieved the reduction in time, the difference increased linearly with the data size. To reproduce 
this test, I created a python script called `generate_plots.py`, the plots are stored in the folder `plots`. Here are the plots to each 
function. In all the plots, the blue line corresponds to the original code.

#### 1. Parse bathroom text to integer
![parse_bathroom](/solution/plots/bathroom.png)

#### 2. Extract amenities
This function extracts the different amenities, while the time is close to the implemented function the code
could easily lead to error due to the copy/paste of the same code with different columns, my implementation only relies on
a list with the different amenities to extract. This code while tested sometimes it did outperform the original code, either way, I 
think that the new implementation is better to code maintenance.

![Amenities](/solution/plots/cat_encoder.png)

#### 3. Pandas cut function

The numpy implementation is similar in code complexity but pandas cut is easier to understand so I kept this part as it is.


![Pandas_cut](/solution/plots/pd_cut.png)

#### 4. Parse the string price to int
![Parse_price](/solution/plots/price.png)

Note: Some of this conclusions may vary slightly if the code is executed in docker or local.

In addition, there are two scripts dedicated to test the original implementation with mine, one for each notebook. This tests can 
be found in the the path `code/test/develope_test` the `test_eda.py` compares the notebook `01-experatory-data-analysis.ipynb` and 
the  `test_explore_classifier.py` compares the notebook `02-explore-classifier-model.ipynb`. The results of this code can be seen in 
the logs folder `test_eda.log` abd `test_explorer.log` respectibely.

The result for the first one is always a bit worse than the original code. This is due to the implementation of the different steps 
of the cleaning a processing process via Skelean `ColumnTransformer` and `Pipelines` the fitting method add some extra time to 
compute the result. Nevertheless, I think that this delay is worth it because it allows to apply the same preprocessing steps to 
unseen data which is usually desired when trying the generated model in unseen data.

In regards to the seccond script, the execution time es slightly better in the refactored code, but the difference in the implementation is very small.

### Maintainability 

As mentioned before, to improve maintainability, I decided to implement the various steps as separate custom column transformers using 
`sklearn`. This approach allows for easier modification of the process and the addition of new steps. The different transformers are 
saved in `code/src/transformer.py`. Unit tests are also included to ensure that the code behaves as expected. By using pipelines, the 
entire process can be summarized as follows:

```python
preprocessing_pipeline = Pipeline(steps=[
        ('col_selector', ColumnSelector(COLUMNS)),
        ('bathroom_processing', StringToFloatTransformer({'bathrooms_text': 'bathrooms'})),
        ('cast_price', StringToInt('price', r"(\d+)")),
        ('filter_rows', QueryFilter("price >= 10")),
        ('drop_na', DropNan(axis=0)),
        ('bin_price', DiscretizerTransformer('price', 'category', bins=[10, 90, 180, 400, np.inf], labels=[0, 1, 2, 3])),
        ('array_to_cat', ArrayOneHotEncoder('amenities', CAT_COLS)),
        ('col_renamer_conditioning', ColumnRenamer(columns={'Air conditioning': 'Air_conditioning', 'neighbourhood_group_cleansed': 'neighbourhood'})),
        ('drop_cols', DropColumns('amenities'))
        ]
    )

ct =  ColumnTransformer(
        [
            ('ordinal_encoder', CustomOrdinalEncoder(categories=[list(MAP_NEIGHB.keys()), list(MAP_ROOM_TYPE.keys())], start_category=1), ["neighbourhood", "room_type"])
        ],
        remainder = "passthrough",
        verbose_feature_names_out=False
    )

processing_pipeline = Pipeline(steps=[
        ('drop_na', DropNan(axis=0)),
        ('categorical', ct),
        ('col_selector', ColumnSelector(FEATURE_NAMES + [TARGET_VARIABLE]))
        ]
    )

data_pipeline = Pipeline(steps=[
        ('data_preprocessing', preprocessing_pipeline),
        ('data_processing', processing_pipeline)
    ])
```

To apply all the transformations at once, it is only necessary to call `data_pipeline`. The process is divided in order to facilitate 
the testing of the
different transformations. This could also be implemented by creating different regular Python functions, but, in my opinion, this 
approach is easier to 
understand, export to other environments, and allows the trained transformers to be applied to new data, avoiding data leakage.

The different transformers could probably be improved or even merged for a cleaner implementation of the transformations. However, I tried to focus more 
on the whole solution rather than aiming for an excellent transformation code, as that part is easier to fix.

### Code testeable

To make the code testable, I separated the different stages of development into different scripts as already explained above. I also 
added unit tests for the transformers to ensure that the results remain correct after changes. And the tests for the results from the 
original code are usefull to check debiations in the global result.

To facilitate the use of the code in different stages within CI, I divided the cleaning process into different pipelines according 
to the notebooks. These pipelines are saved using joblib to make them reusable. Additionally, I deployed an `MLflow` instance to 
save the model and the pipeline using the `MLflow.Pyfunc` class for the entire pipeline, the processing pipeline, and the trained 
model. This makes it easier to use this code in the API, avoiding issues with the environment, code changes, or updates in the 
models themselves.

## Challenge 2 - Build an API


To implement the API, I used the `FastAPI` framework along with Pydantic for validation of input/output data. The API is hosted 
locally on `localhost:8000`. FastAPI includes an automatically generated documentation interface, `http://localhost:8000/`, where 
example calls can be tested interactively. 

The primary endpoint for this API can be accessed programmatically at `http://localhost:8000/model-inference`. The expected input 
and ouput matches the format in the README file. Additionally, the endpoint also supports an array of elements, provided all 
elements have the same length and adhere to the defined input schema. Here an example calling the endpoint programatically:

```python
import requests

payload = {
  "accommodates": [4, 4],
  "bathrooms": [2,2],
  "bedrooms": [1,1],
  "beds": [2,2],
  "elevator": [1,1],
  "id": [1001,1001],
  "internet": [0,0],
  "latitude": [40.71383, 456],
  "longitude": [-73.9658, 56],
  "neighbourhood": ["Brooklyn", "Brooklyn"],
  "room_type": ["Entire home/apt", "Entire home/apt"], 
  "tv": [1, 1]
}
response = requests.post("http://localhost:8080/model-inference", json=payload)
response.json()
# expected output
{'id': [1001, 1001], 'price_category': ['High', 'High']}
```

## Challenge 3 - Dockerize your solution

To dockerize the solution I used Docker Compose with three Docker Images:
-	**App**: Which creates the endpoint for the API to get the predictions.
-	**Mlflow**: Which creates a server to save and load the model without copying the enviroment from one place to another.
-	**Pipeline**: This image contains all the code explained before, it saves the models, the logs and the plots. This image takes a bit of time due to the testing of bigger sample data for the plots.

There is alson a `.env` file to store the endpoint of MLFlow in the other images to grant conectibity to the MLFlow server. To deploy 
the solution it is only necessary to run `docker compose up --build` in the docker directory and wait arround one minute to have 
everything ready.


Note: To run the different scripts locally, execute the code from the solution folder as follows:
```bash
PYTHONPATH="${PYTHONPATH}:../" python code/generate_plots.py
PYTHONPATH="${PYTHONPATH}:../" python code/pipeline.py
PYTHONPATH="${PYTHONPATH}:../" python code/test/test_transformers.py 
PYTHONPATH="${PYTHONPATH}:../" python code/test/develope_tests/test_eda.py
PYTHONPATH="${PYTHONPATH}:../" python code/test/test_transformers.py 
```