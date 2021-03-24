import pickle
import config

def save_object(filename, obj):
    
    if filename is None:
        print("Please provide filename.")
    
    f = open(config.DATA_DIR+filename, 'wb')
    pickle.dump(obj, f)
    f.close()

def load_object(filename):
    
    f = open(config.DATA_DIR+filename, 'rb')
    loaded = pickle.load(f)
    f.close()
        
    return loaded
