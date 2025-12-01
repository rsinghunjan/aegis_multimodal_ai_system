class AegisError(Exception):
    pass

class ModelLoadingError(AegisError):
    pass

class DataLoadingError(AegisError):
    pass

# Example helper to wrap exceptions (keeps semantics explicit)
def wrap_model_loading(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise ModelLoadingError(str(e)) from e
    return wrapper
