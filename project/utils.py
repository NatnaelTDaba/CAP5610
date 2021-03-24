import pickle
import config

def save_object(filename, obj):

    """
        Args:
            filename (string): Name that the saved file should take
            obj (object): Object to be saved
    """
    if filename is None:
        print("Please provide filename.")
    
    f = open(config.DATA_DIR+filename, 'wb')
    pickle.dump(obj, f)
    f.close()

def load_object(filename):

     """
        Args:
            filename (string): Name file to be loaded

        Returns: loaded object

    """
    
    f = open(config.DATA_DIR+filename, 'rb')
    loaded = pickle.load(f)
    f.close()
        
    return loaded
