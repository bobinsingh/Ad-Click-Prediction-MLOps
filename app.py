from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse
from uvicorn import run as app_run

from typing import Optional

# Importing constants and pipeline modules from the project
from src.constants import APP_HOST, APP_PORT
from src.pipelines.predict_pipeline import AdData, AdDataClassifier
from src.pipelines.train_pipeline import TrainPipeline

# Initialize FastAPI application
app = FastAPI()

# Mount the 'static' directory for serving static files (like CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 template engine for rendering HTML templates
templates = Jinja2Templates(directory='templates')

# Allow all origins for Cross-Origin Resource Sharing (CORS)
origins = ["*"]

# Configure middleware to handle CORS, allowing requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DataForm:
    """
    DataForm class to handle and process incoming form data.
    This class defines the ad-related attributes expected from the form.
    """
    def __init__(self, request: Request):
        self.request: Request = request
        self.age: Optional[int] = None
        self.gender_Male: Optional[int] = None
        self.gender_Non_Binary: Optional[int] = None
        self.device_type_Mobile: Optional[int] = None
        self.device_type_Tablet: Optional[int] = None
        self.ad_position_Side: Optional[int] = None
        self.ad_position_Top: Optional[int] = None
        self.browsing_history_Entertainment: Optional[int] = None
        self.browsing_history_News: Optional[int] = None
        self.browsing_history_Shopping: Optional[int] = None
        self.browsing_history_Social_Media: Optional[int] = None
        self.time_of_day_Evening: Optional[int] = None
        self.time_of_day_Morning: Optional[int] = None
        self.time_of_day_Night: Optional[int] = None

    async def get_user_data(self):
        """
        Method to retrieve and assign form data to class attributes.
        This method is asynchronous to handle form data fetching without blocking.
        """
        form = await self.request.form()
        self.age = form.get("age")
        self.gender_Male = form.get("gender_Male")
        self.gender_Non_Binary = form.get("gender_Non_Binary")
        self.device_type_Mobile = form.get("device_type_Mobile")
        self.device_type_Tablet = form.get("device_type_Tablet")
        self.ad_position_Side = form.get("ad_position_Side")
        self.ad_position_Top = form.get("ad_position_Top")
        self.browsing_history_Entertainment = form.get("browsing_history_Entertainment")
        self.browsing_history_News = form.get("browsing_history_News")
        self.browsing_history_Shopping = form.get("browsing_history_Shopping")
        self.browsing_history_Social_Media = form.get("browsing_history_Social_Media")
        self.time_of_day_Evening = form.get("time_of_day_Evening")
        self.time_of_day_Morning = form.get("time_of_day_Morning")
        self.time_of_day_Night = form.get("time_of_day_Night")


# Route to render the main page with the form
@app.get("/", tags=["authentication"])
async def index(request: Request):
    """
    Renders the main HTML form page for ad data input.
    """
    return templates.TemplateResponse(
            "addata.html", {"request": request, "context": "Rendering"})


# Route to trigger the model training process
@app.get("/train")
async def trainRouteClient():
    """
    Endpoint to initiate the model training pipeline.
    """
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        return Response("Training successful!!!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")


# Route to handle form submission and make predictions
@app.post("/")
async def predictRouteClient(request: Request):
    """
    Endpoint to receive form data, process it, and make a prediction.
    """
    try:
        form = DataForm(request)
        await form.get_user_data()
        
        # Prepare data based on form input
        ad_data = AdData(
            age=form.age,
            gender_Male=form.gender_Male,
            gender_Non_Binary=form.gender_Non_Binary,
            device_type_Mobile=form.device_type_Mobile,
            device_type_Tablet=form.device_type_Tablet,
            ad_position_Side=form.ad_position_Side,
            ad_position_Top=form.ad_position_Top,
            browsing_history_Entertainment=form.browsing_history_Entertainment,
            browsing_history_News=form.browsing_history_News,
            browsing_history_Shopping=form.browsing_history_Shopping,
            browsing_history_Social_Media=form.browsing_history_Social_Media,
            time_of_day_Evening=form.time_of_day_Evening,
            time_of_day_Morning=form.time_of_day_Morning,
            time_of_day_Night=form.time_of_day_Night
        )

        # Convert form data into a DataFrame for the model
        ad_df = ad_data.get_ad_input_data_frame()

        # Initialize the prediction pipeline
        model_predictor = AdDataClassifier()

        # Make a prediction and retrieve the result
        value = model_predictor.predict(dataframe=ad_df)[0]

        # Interpret the prediction result as 'Response-Yes' or 'Response-No'
        status = "User Will Click Ad" if value == 1 else "User Will Not Click Ad"

        # Render the same HTML page with the prediction result
        return templates.TemplateResponse(
            "addata.html",
            {"request": request, "context": status},
        )
        
    except Exception as e:
        return {"status": False, "error": f"{e}"}


# Main entry point to start the FastAPI server
if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)



# from fastapi import FastAPI, Request
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import Response
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates
# from starlette.responses import HTMLResponse
# from uvicorn import run as app_run

# from typing import Optional

# # Importing constants and pipeline modules from the project
# from src.constants import APP_HOST, APP_PORT
# from src.pipelines.predict_pipeline import AdData, AdDataClassifier
# from src.pipelines.train_pipeline import TrainPipeline

# # Initialize FastAPI application
# app = FastAPI()

# # Mount the 'static' directory for serving static files (like CSS)
# app.mount("/static", StaticFiles(directory="static"), name="static")

# # Set up Jinja2 template engine for rendering HTML templates
# templates = Jinja2Templates(directory='templates')

# # Allow all origins for Cross-Origin Resource Sharing (CORS)
# origins = ["*"]

# # Configure middleware to handle CORS, allowing requests from any origin
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class DataForm:
#     """
#     DataForm class to handle and process incoming form data.
#     This class defines the ad-related attributes expected from the form.
#     """
#     def __init__(self, request: Request):
#         self.request: Request = request
#         self.gender_Male: Optional[int] = None
#         self.gender_Non_Binary: Optional[int] = None
#         self.gender_Unknown: Optional[int] = None
#         self.age: Optional[int] = None
#         self.device_type_Mobile: Optional[int] = None
#         self.device_type_Tablet: Optional[int] = None
#         self.device_type_Unknown: Optional[int] = None
#         self.ad_position_Side: Optional[int] = None
#         self.ad_position_Top: Optional[int] = None
#         self.ad_position_Unknown: Optional[int] = None
#         self.browsing_history_Entertainment: Optional[int] = None
#         self.browsing_history_News: Optional[int] = None
#         self.browsing_history_Shopping: Optional[int] = None
#         self.browsing_history_Social_Media: Optional[int] = None
#         self.browsing_history_Unknown: Optional[int] = None
#         self.time_of_day_Evening: Optional[int] = None
#         self.time_of_day_Morning: Optional[int] = None
#         self.time_of_day_Night: Optional[int] = None
#         self.time_of_day_Unknown: Optional[int] = None

#     async def get_user_data(self):
#         """
#         Method to retrieve and assign form data to class attributes.
#         This method is asynchronous to handle form data fetching without blocking.
#         """
#         form = await self.request.form()
#         self.gender_Male = form.get("gender_Male")
#         self.gender_Non_Binary = form.get("gender_Non_Binary")
#         self.gender_Unknown = form.get("gender_Unknown")
#         self.age = form.get("age")
#         self.device_type_Mobile = form.get("device_type_Mobile")
#         self.device_type_Tablet = form.get("device_type_Tablet")
#         self.device_type_Unknown = form.get("device_type_Unknown")
#         self.ad_position_Side = form.get("ad_position_Side")
#         self.ad_position_Top = form.get("ad_position_Top")
#         self.ad_position_Unknown = form.get("ad_position_Unknown")
#         self.browsing_history_Entertainment = form.get("browsing_history_Entertainment")
#         self.browsing_history_News = form.get("browsing_history_News")
#         self.browsing_history_Shopping = form.get("browsing_history_Shopping")
#         self.browsing_history_Social_Media = form.get("browsing_history_Social_Media")
#         self.browsing_history_Unknown = form.get("browsing_history_Unknown")
#         self.time_of_day_Evening = form.get("time_of_day_Evening")
#         self.time_of_day_Morning = form.get("time_of_day_Morning")
#         self.time_of_day_Night = form.get("time_of_day_Night")
#         self.time_of_day_Unknown = form.get("time_of_day_Unknown")


# # Route to render the main page with the form
# @app.get("/", tags=["authentication"])
# async def index(request: Request):
#     """
#     Renders the main HTML form page for ad data input.
#     """
#     return templates.TemplateResponse(
#             "addata.html", {"request": request, "context": "Rendering"})


# # Route to trigger the model training process
# @app.get("/train")
# async def trainRouteClient():
#     """
#     Endpoint to initiate the model training pipeline.
#     """
#     try:
#         train_pipeline = TrainPipeline()
#         train_pipeline.run_pipeline()
#         return Response("Training successful!!!")

#     except Exception as e:
#         return Response(f"Error Occurred! {e}")


# # Route to handle form submission and make predictions
# @app.post("/")
# async def predictRouteClient(request: Request):
#     """
#     Endpoint to receive form data, process it, and make a prediction.
#     """
#     try:
#         form = DataForm(request)
#         await form.get_user_data()

#         ad_data = AdData(
#             gender_Male=form.gender_Male,
#             gender_Non_Binary=form.gender_Non_Binary,
#             gender_Unknown=form.gender_Unknown,
#             age=form.age,
#             device_type_Mobile=form.device_type_Mobile,
#             device_type_Tablet=form.device_type_Tablet,
#             device_type_Unknown=form.device_type_Unknown,
#             ad_position_Side=form.ad_position_Side,
#             ad_position_Top=form.ad_position_Top,
#             ad_position_Unknown=form.ad_position_Unknown,
#             browsing_history_Entertainment=form.browsing_history_Entertainment,
#             browsing_history_News=form.browsing_history_News,
#             browsing_history_Shopping=form.browsing_history_Shopping,
#             browsing_history_Social_Media=form.browsing_history_Social_Media,
#             browsing_history_Unknown=form.browsing_history_Unknown,
#             time_of_day_Evening=form.time_of_day_Evening,
#             time_of_day_Morning=form.time_of_day_Morning,
#             time_of_day_Night=form.time_of_day_Night,
#             time_of_day_Unknown=form.time_of_day_Unknown
#         )

#         # Convert form data into a DataFrame for the model
#         ad_df = ad_data.get_ad_input_data_frame()

#         # Initialize the prediction pipeline
#         model_predictor = AdDataClassifier()

#         # Make a prediction and retrieve the result
#         value = model_predictor.predict(dataframe=ad_df)[0]

#         # Interpret the prediction result as 'Response-Yes' or 'Response-No'
#         status = "User Will Click Ad" if value == 1 else "User Will Not Click Ad"

#         # Render the same HTML page with the prediction result
#         return templates.TemplateResponse(
#             "addata.html",
#             {"request": request, "context": status},
#         )

#     except Exception as e:
#         return {"status": False, "error": f"{e}"}


# # Main entry point to start the FastAPI server
# if __name__ == "__main__":
#     app_run(app, host=APP_HOST, port=APP_PORT)
