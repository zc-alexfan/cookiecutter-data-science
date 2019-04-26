import glob


def get_image_indices(folder, ext = ".jpg"): 
    files = glob.glob(folder +"*" + ext)  
    files = [int(f.replace(folder, '').replace(ext, '')) for f in files]
    files.sort()
    if(len(files) == 0):
        print("Warning: No image indices are found in folder (%s)"%(folder))
    return files


def get_my_logger(logger_name, print_console=True, print_file=True):
    import logging, os
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s', "%m-%d %H:%M:%S")

    if print_file:
        file_handler = logging.FileHandler(os.path.join(cfg.log_path, '%s.log'%(logger_name)))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if print_console:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    logger.info(">>>>>>>>>START logger<<<<<<<<<<<")
    return logger

    