import os
import sys
import json
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from copy import deepcopy as dp

import monai
import torch
import pytorch_lightning
import monai.data as monaiData
import monai.metrics as monaiMetrics
import monai.transforms as monaiTrans
from torch.utils.data import DataLoader as torchDataloader


import biswebpython.utilities.modelObjects as model_objects

from monai.inferers import sliding_window_inference





class TRANSFORMATION:

    def __init__(self):
        self.index = 4
        self.comfuncs = []
        self.keys = []
        self.orientation = 'LPS'
        self.patch_size = None
        self.spacing = None
        self.prefix = None
        self.Spacingd = {
            'mode': ('bilinear', 'nearest'),
            'padding_mode': ('reflection','reflection')
        }
        self.ScaleIntensityRangePercentilesd = {
            'keys': ['IMAGE'],
            'lower': 25,
            'upper': 75,
            'b_min': -0.5,
            'b_max': 0.5,
            'clip': False
        }
        self.RandCropByPosNegLabeld = {
            'label_key': 'SEGM',
            'pos': 3,
            'neg': 1,
            'num_samples': 2
        }


    def init_comfuncs(self, status):

        self.comfuncs = [
            monaiTrans.LoadNiftid(keys=self.keys),
            monaiTrans.AddChanneld(keys=self.keys),
            monaiTrans.Orientationd(keys=self.keys, axcodes=self.orientation),
            monaiTrans.Spacingd(keys=self.keys, pixdim=self.spacing, mode=self.Spacingd['mode'], padding_mode=self.Spacingd['padding_mode']),
            monaiTrans.ScaleIntensityRangePercentilesd(keys=self.ScaleIntensityRangePercentilesd['keys'], \
                                                       lower=self.ScaleIntensityRangePercentilesd['lower'], upper=self.ScaleIntensityRangePercentilesd['upper'], \
                                                       b_min=self.ScaleIntensityRangePercentilesd['b_min'], b_max=self.ScaleIntensityRangePercentilesd['b_max'], \
                                                       clip=self.ScaleIntensityRangePercentilesd['clip']),
            monaiTrans.ToTensord(keys=self.keys)
        ]

        if status == 'train':
            # randomly crop out patch samples from big image based on pos / neg ratio
            # the image centers of negative samples must be in valid image area
            self.comfuncs.insert(-1, monaiTrans.RandCropByPosNegLabeld(keys=self.keys, label_key=self.RandCropByPosNegLabeld['label_key'], \
                                              spatial_size=self.patch_size, num_samples=self.RandCropByPosNegLabeld['num_samples'], \
                                              pos=self.RandCropByPosNegLabeld['pos'], neg=self.RandCropByPosNegLabeld['neg']),)


    def parsingtransformations(self, status, debug, dt_dict=None, aug_dict=None):

        if dt_dict:
            if 'orientation' in dt_dict.keys():
                self.orientation = dt_dict['orientation'].upper()

            try:
                # TODO: 2D framework---------------------------------------------------------------------------------------------------------
                self.patch_size = tuple(dt_dict['patch_size'])
                # TODO: 2D framework---------------------------------------------------------------------------------------------------------

                self.spacing = tuple(dt_dict['spacing'])
                if len(self.keys) == 1:
                    self.Spacingd['mode'] = self.Spacingd['mode'][0]
                    self.Spacingd['padding_mode'] = self.Spacingd['padding_mode'][0]
            except:
                print("patch_size / spacing must be defined in the json file. Please double check." )
                raise ValueError

            fix_keys = ['orientation', 'spacing', 'patch_size']
            for eledf in dt_dict.keys():
                if eledf not in fix_keys:
                    for argd in dt_dict[eledf].keys():
                        vars(self)[eledf][argd] = dt_dict[eledf][argd]

        self.init_comfuncs(status)



        if aug_dict:
            for elef in aug_dict.keys():

                try:
                    new_dict = dp(aug_dict[elef])
                    new_dict['functionname'] = elef
                    sgfunc = model_objects.concatFunctionArguments(new_dict, prefix='monaiTrans.')

                    if 'keys' not in aug_dict[elef].keys():
                        sgfunc = sgfunc[:sgfunc.index('(')+1] + 'keys=self.keys, ' +sgfunc[sgfunc.index('(')+1:]

                    self.comfuncs.insert(self.index, eval(sgfunc))
                    self.index += 1

                except:
                    print("Cannot apply the transformation, "+ elef +", on the datasest. Please double check." )
                    raise ValueError



        if debug:
            statics = 'monaiTrans.DataStatsd(keys=' + str(self.keys) +', prefix=' + str(self.prefix) + ')'
            # self.comfuncs.insert(3, eval(statics))
            self.comfuncs.insert(-2, eval(statics))






def convertInputsToDictionaies(PATH, debug):
    df = pd.read_csv(PATH, index_col=False)
    df.drop_duplicates(inplace=True)
    df = df[['IMAGE','SEGM','DATA_SPLIT']]

    df_train = df[df['DATA_SPLIT']=='Training']
    df_val = df[df['DATA_SPLIT']=='Validation']
    df_test = df[df['DATA_SPLIT']=='Testing']

    # Convert DF to dictionary
    train_dict = df_train.to_dict('records')
    val_dict = df_val.to_dict('records')
    test_dict = df_test.to_dict('records')


    if debug:
        print('Dataset contains %d entries' % len(df))
        print('Number of training files: ', len(train_dict))
        print('Number of validation files: ', len(val_dict))
        print('Number of testing files: ', len(test_dict))

    return train_dict, val_dict, test_dict





def initParams():
    default_parameters = {}
    default_parameters['model'] = {}


    # initiate parameters for initiating the data loading
    default_parameters['batch_size'] = 16
    default_parameters['shuffle'] = True
    default_parameters['num_workers'] = 4
    default_parameters['collate_fn'] = monaiData.list_data_collate


    # initiate parameters for training
    default_parameters['max_epochs'] = 2000
    default_parameters['num_sanity_val_steps'] = 0
    default_parameters['check_val_every_n_epoch'] = 10


    # initiate parameters for initiating the model
    default_parameters['model']['modelname'] = 'UNet'
    default_parameters['model']['dimensions'] = 3
    default_parameters['model']['in_channels'] = 1
    default_parameters['model']['out_channels'] = 2
    default_parameters['model']['channels'] = (16, 32, 64, 128)
    default_parameters['model']['stride_size'] = 2
    default_parameters['model']['strides'] = None
    default_parameters['model']['num_res_units'] = 2
    default_parameters['model']['roi_size'] = None
    default_parameters['model']['sw_batch_size'] = 4
    default_parameters['model']['lossfunction'] = {
        'functionname': 'DiceLoss',
        'to_onehot_y': True,
        'softmax': True
    }
    default_parameters['model']['metricfunction'] = {
        'functionname': 'DiceMetric',
        'include_background': True,
        'to_onehot_y': True,
        'sigmoid': True,
        'reduction': 'mean'
    }
    default_parameters['model']['optimfunction'] = {
        'functionname': 'Adam',
        'lr': 1e-4
    }

    default_parameters['testSegm'] = True
    default_parameters['postprocessing'] = True
    default_parameters['postprocessing_func'] = {
        'functionname': 'KeepLargestConnectedComponent',
        'applied_labels': [1]
    }

    return default_parameters





def updateParams(inputs, defaults, status):

    if status == 'model':
        for model_arg in inputs.keys():
            if model_arg == 'name':
                if inputs['name'].lower() == 'unet3d':
                    defaults['model']['modelname'] == 'UNet'
                    defaults['model']['dimensions'] == 3
                elif inputs['name'].lower() == 'unet2d':
                    defaults['model']['modelname'] == 'UNet'
                    defaults['model']['dimensions'] == 2
            else:
                defaults['model'][model_arg] = inputs[model_arg]
        if not defaults['model']['strides']:
            defaults['model']['strides'] = tuple([defaults['model']['stride_size'] for i in range(len(defaults['model']['channels'])-1)])



    else:
        if status == 'validate' or status == 'test':
            defaults['batch_size'] = 1
            defaults['shuffle'] = False

        for sarg in inputs.keys():
            defaults[sarg] = inputs[sarg]








def parsingInputs(inps, debug):
    CACHE_PATH = None
    INPUT_PATH = inps['inputpath']
    if 'cachepath' in inps.keys():
        CACHE_PATH = inps['cachepath']


    # Load dataset and convert to dictionary
    traind, validated, testd = convertInputsToDictionaies(INPUT_PATH, debug)
    res_loader = {}
    params = initParams()
    params['debug'] = debug
    updateParams(inps['model'], params, 'model')

    statuslist = []
    if 'train' in inps.keys():
        statuslist.append('train')
        # TODO: NO VALIDATION DATA ----------------------------------------------------------------
        if validated == []:
            params['val_percent_check']=0
        else:
            statuslist.append('validate')
    if 'test' in inps.keys():
        statuslist.append('test')


    for status in statuslist:
        input_defaultT = None
        input_augmentation = None

        # Parsing transformation parameters
        vars()[status+'_transformation'] = TRANSFORMATION()
        if status == 'test' and type(testd[0]['SEGM']) == float:
            params['testSegm'] = False
            vars()[status+'_transformation'].keys = ['IMAGE']
            vars()[status+'_transformation'].prefix = (status + '_image',)
        else:
            vars()[status+'_transformation'].keys = ['IMAGE', 'SEGM']
            vars()[status+'_transformation'].prefix = (status + '_image', status + '_segm')


        if status in inps.keys() and 'defaulttransformation' in inps[status].keys():
            input_defaultT = inps[status].pop('defaulttransformation')
        elif 'defaulttransformation' in inps.keys():
            input_defaultT = inps['defaulttransformation']
        if status in inps.keys() and 'augmentation' in inps[status].keys():
            input_augmentation = inps[status].pop('augmentation')
        elif 'augmentation' in inps.keys():
            input_augmentation = inps['augmentation']

        vars()[status+'_transformation'].parsingtransformations(status, debug, input_defaultT, input_augmentation)

        trans_lists = getattr(vars()[status+'_transformation'], 'comfuncs')


        # Parsing status specific arguments
        if status in inps.keys():
            updateParams(inps[status], params, status)
        else:
            updateParams({}, params, status)


        # Maintain a consistent CACHE_PATH if you want mulitple programs to use this
        vars()[status+'_ds'] = monaiData.PersistentDataset(
            data = vars()[status+'d'], \
            transform = monaiTrans.Compose(trans_lists), \
            cache_dir=CACHE_PATH\
        )
        vars()[status+'_loader'] = torchDataloader(
            vars()[status+'_ds'], \
            batch_size = params['batch_size'], \
            shuffle =params['shuffle'], \
            num_workers =params['num_workers'], \
            collate_fn = params['collate_fn']\
        )

        res_loader[status+'_loader'] = vars()[status+'_loader']


    # TODO: user should keep the parameteres below consistent in training / testing
    try:
        params['model']['patch_size'] = vars()['train_transformation'].patch_size
        params['orientation'] = vars()['train_transformation'].orientation
        params['spacing'] = vars()['train_transformation'].spacing
    except:
        pass
    try:
        params['orientation'] = vars()['test_transformation'].orientation
        params['spacing'] = vars()['test_transformation'].spacing
    except:
        pass

    return params, res_loader







def findMaxDim(loaders):
    # TODO: test with images with diferent shape----------------------------------------------
    max_dims = []

    with torch.no_grad():
        for key in loaders.keys():
            loader = loaders[key]
            for i, data in enumerate(loader):
                image_dims = data['IMAGE_meta_dict']['spatial_shape'].detach().clone().cpu().numpy()
                if i == 0:
                    temp_max = np.amax(image_dims, axis=0)
                else:
                    temp_max = np.amax([temp_max, np.amax(image_dims, axis=0)], axis=0)
            max_dims.append(temp_max)
    max_dim = list(np.amax(max_dims, axis=0))
    return max_dim






def initRoisize(loaders, min_num):
    NEW_DIM = []
    DIM = findMaxDim(loaders)

    for idx in range(len(DIM)):
        dim = DIM[idx]

        if (dim % min_num):
            new_dim = (dim // min_num + 1) * min_num
            NEW_DIM.append(new_dim)
        else:
            NEW_DIM.append(dim)

    return tuple(NEW_DIM)






def setupLoggers(MODEL_ROOT_PATH):
    logger = pytorch_lightning.loggers.TensorBoardLogger(
        save_dir=os.path.join(MODEL_ROOT_PATH,'saved_model','logs')
    )

    callback = pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint(
        filepath=os.path.join(MODEL_ROOT_PATH, 'saved_model',"{epoch}-{val_dice:.2f}"),
        save_last=True,
        save_top_k=3,
    )

    # check for last checkpoint
    lastCheckpoint = None
    if os.path.exists(os.path.join(MODEL_ROOT_PATH,'saved_model','last.ckpt')):
        lastCheckpoint = os.path.join(MODEL_ROOT_PATH,'saved_model','last.ckpt')

    return [logger, callback, lastCheckpoint]





def initTrainer(logL, gpuidx, dattrs):
    trainer = pytorch_lightning.Trainer(
        gpus=[gpuidx],
        logger=logL[0],
        checkpoint_callback=logL[1],
        resume_from_checkpoint=logL[2]
    )

    for key in dattrs.keys():
        # if hasattr(trainer, key):
        if key in trainer.default_attributes():
            vars(trainer)[key] = dattrs[key]
            # TODO: val_percent_check cannnot be assigned with vars()--------------------------------------
            # TODO: fix with the following if statement--------------------------------------
            if key == 'val_percent_check':
                trainer.val_percent_check = 0.0

    return trainer






def getAffineMatrix(orientation, spacing, dimension):
    affine = np.eye(dimension+1)
    for i in range(len(spacing)):
        affine[i,i] = spacing[i]

    if dimension == 3:
        if orientation[0] == 'L':
            affine[0, 0] *=  -1
        if orientation[1] == 'P':
            affine[1, 1] *=  -1
        if orientation[2] == 'I':
            affine[2, 2] *=  -1
    else:
        # TODO: xenos please take a look at it---------------------------------------------
        pass

    return affine






def predictionAndEvaluation(params, test_loader, device, model):

    # TODO: affine matrices with different orientation -------------------------------------------------------------------------------
    # TODO: 2D affine 'ra' or 'rs'
    # TODO: respacing ---------------------------------
    test_affine = getAffineMatrix(params['orientation'], params['spacing'], params['model']['dimensions'])



    model_metrics = params['model']['metricfunction']
    model_metrics['reduction'] = 'none'
    eval_metric = eval(model_objects.concatFunctionArguments(model_metrics, prefix='monaiMetrics.'))

    original_dice_results = list()
    postprocess_dice_results = list()
    output_results = list()

    original_hd_results = list()
    postprocess_hd_results = list()

    original_mad_results = list()
    postprocess_mad_results = list()


    with torch.no_grad():
        for i, test_data in enumerate(test_loader):
            roi_size = params['model']['roi_size']
            sw_batch_size = params['model']['sw_batch_size']

            test_outputs = sliding_window_inference(test_data['IMAGE'].to(device), roi_size, sw_batch_size, model)
            argmax = torch.argmax(test_outputs, dim=1, keepdim=True)


            # Post-processing
            if params['postprocessing']:
                post_func = eval(model_objects.concatFunctionArguments(params['postprocessing_func'], prefix='monaiTrans.'))
                largest = post_func(argmax)
            else:
                largest = argmax


            # Write data out
            output_file_name = os.path.split(test_loader.dataset.data[i]['IMAGE'])[1]
            output_path = os.path.join(params['output_path'],output_file_name)
            output_results.append(output_path)

            test_data_nii = nib.load(test_loader.dataset.data[i]['IMAGE'])
            target_affine = test_data_nii.affine

            # Remove the translation component
            # TODO: 2D-----------------------------------------------------------------------------------------
            # TODO: xenos please take a look at it
            target_affine[0:params['model']['dimensions'],params['model']['dimensions']] = 0
            # TODO: xenos please take a look at it: resampling ------------------------------------------------------------------
            monaiData.write_nifti(largest.detach().cpu()[0, 0,...].numpy(), output_path,
                        mode='nearest',
                        affine=test_affine,
                        target_affine=target_affine,
                        output_spatial_shape=test_data_nii.shape,
                        dtype=np.float32
                       )

            if params['debug']:
                print(test_data['IMAGE'].shape)
                print(output_path)



            # evaluation scores
            if params['testSegm']:
                test_labels = test_data['SEGM'].to(device)


                value = eval_metric(y_pred=argmax, y=test_labels)
                print('Dice: {:.5f}'.format(value.item()))
                original_dice_results.append(value.item())

                hd_value = monaiMetrics.compute_hausdorff_distance(argmax, test_labels, label_idx=1, percentile=95)
                print('HD95: {:.5f}'.format(hd_value.item()))
                original_hd_results.append(hd_value)

                mad_value = monaiMetrics.compute_average_surface_distance(argmax, test_labels, label_idx=1)
                print('MAD: {:.5f}'.format(mad_value.item()))
                original_mad_results.append(mad_value)

                if params['postprocessing']:
                    value = eval_metric(y_pred=largest, y=test_labels)
                    postprocess_dice_results.append(value.item())
                    print('Post-processed Dice: {:.5f}'.format(value.item()))

                    hd_value = monaiMetrics.compute_hausdorff_distance(largest, test_labels, label_idx=1, percentile=95)
                    print('Post-processed HD95: {:.5f}'.format(hd_value.item()))
                    postprocess_hd_results.append(hd_value)

                    mad_value = monaiMetrics.compute_average_surface_distance(largest, test_labels, label_idx=1)
                    print('Post-processed HD95: {:.5f}'.format(mad_value.item()))
                    postprocess_mad_results.append(mad_value)



    if params['testSegm']:
        eval_results = pd.DataFrame()

        eval_results['IMAGE_DATA'] = [ i['IMAGE'] for i in test_loader.dataset.data]
        eval_results['SEGM_DATA'] = [ i['SEGM'] for i in test_loader.dataset.data]
        eval_results['SEGM_RESULTS'] = output_results

        eval_results['DICE'] = original_dice_results

        if params['postprocessing']:
            eval_results['POST_DICE'] = postprocess_dice_results

        eval_results['HD95'] = original_hd_results
        if params['postprocessing']:
            eval_results['POST_HD95'] = postprocess_hd_results

        eval_results['MAD'] = original_mad_results
        if params['postprocessing']:
            eval_results['POST_MAD'] = postprocess_mad_results


        eval_results.to_csv(os.path.join(params['output_path'],'evaluation_results.csv'), index=False)








def imageSegmentation(paramfile, debug):



    # ## Verify System Setup
    # Check torch and CUDA on the system.
    if debug:
        monai.config.print_config()

        print('CUDA available: ', torch.cuda.is_available())
        n_gpus = torch.cuda.device_count()
        for i in range(n_gpus):
            print('GPU %d: %s' % (i, torch.cuda.get_device_name(i)))


    # ## Parsing input parameters
    _defaults, loaders = parsingInputs(paramfile, debug)

    if _defaults['model']['roi_size'] == None:
        _defaults['model']['roi_size'] = initRoisize(loaders, np.prod(_defaults['model']['strides']))

    # ## initialize training model in training mode
    MODEL = vars(model_objects)[_defaults['model']['modelname']]( _defaults['model'])
    if debug:
        print(_defaults['model']['modelname'], " MODEL STRUCTURE:")
        print(MODEL)

    if 'train' in paramfile.keys():

        # set up loggers and checkpoints
        loggers_list = setupLoggers(paramfile['outputmodelpath'])

        # # initialise Lightning's trainer.
        TRAINER = initTrainer(loggers_list, paramfile['gpu_device'], _defaults)

        # train
        if 'validate_loader' in loaders.keys():
            TRAINER.fit(MODEL, train_dataloader=loaders['train_loader'], val_dataloaders=loaders['validate_loader'])
        else:
            TRAINER.fit(MODEL, train_dataloader=loaders['train_loader'])

    if 'test' in paramfile.keys():

        # load the last checkpoint of the trained model
        lastcheckpoint = torch.load(os.path.join(paramfile['outputmodelpath'],'saved_model','last.ckpt'))
        MODEL.load_state_dict(lastcheckpoint['state_dict'])


        OUTPUT_PATH = os.path.join(paramfile['outputmodelpath'],'results')
        Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)


        device = torch.device("cuda:"+str(paramfile['gpu_device']))
        MODEL.to(device)


        _defaults['output_path'] = OUTPUT_PATH
        predictionAndEvaluation(_defaults, loaders['test_loader'], device, MODEL)
