import os
from pathlib import Path

project_name = "src"

list_of_files = [
    # Project core structure
    f"{project_name}/__init__.py",
    f"{project_name}/components/__init__.py",
    f"{project_name}/components/data_ingestion.py",  
    f"{project_name}/components/data_validation.py",
    f"{project_name}/components/data_transformation.py",
    f"{project_name}/components/model_trainer.py",
    f"{project_name}/components/model_evaluator.py",  
    f"{project_name}/components/model_deployment.py",  
    f"{project_name}/config/__init__.py",
    f"{project_name}/config/mongo_db_config.py",  
    f"{project_name}/config/mongo_db_handler.py",  

    # Cloud and storage
    f"{project_name}/cloud/__init__.py",
    f"{project_name}/cloud/aws_handler.py",
    f"{project_name}/cloud/aws_storage.py",
    f"{project_name}/cloud/aws_config.py", 
    f"{project_name}/data/__init__.py",  
    f"{project_name}/data/proj_data_handler.py",  

    # Constants and entities
    f"{project_name}/constants/__init__.py",
    f"{project_name}/entities/__init__.py",  
    f"{project_name}/entities/config_entity.py",
    f"{project_name}/entities/artifact_entity.py",
    f"{project_name}/entities/estimator_config.py", 
    f"{project_name}/entities/s3_config.py",  

    # Exception and logging
    f"{project_name}/exceptions/__init__.py",  
    f"{project_name}/logging/__init__.py",  

    # Pipelines
    f"{project_name}/pipelines/__init__.py",
    f"{project_name}/pipelines/train_pipeline.py",  
    f"{project_name}/pipelines/predict_pipeline.py",  

    # Utilities
    f"{project_name}/utils/__init__.py",
    f"{project_name}/utils/helpers.py",
    f"{project_name}/utils/transformation_utils.py",  

    # Testing and documentation
    f"{project_name}/tests/__init__.py",
    f"{project_name}/tests/test_pipeline.py",
    f"{project_name}/docs/Project_flow.md",  

    # Root-level files
    "app.py",
    "requirements.txt",
    "Dockerfile",
    ".dockerignore",
    "setup.py",
    "pyproject.toml",
    ".gitignore", 
    "README.md",  
    "configs/model.yaml",
    "configs/schema.yaml",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
    else:
        print(f"file is already present at: {filepath}")