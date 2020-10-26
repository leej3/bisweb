import torch
import monai
import pytorch_lightning
import monai.losses as LOSS
import monai.metrics as METRIC
import torch.optim as OPTIMIZATION
from monai.networks.layers import Norm
from monai.utils import set_determinism

from monai.inferers import sliding_window_inference




def concatFunctionArguments(dict, prefix=''):

    if len(dict.keys()) == 1 and 'functionname' in dict.keys():
        sfunction = prefix + dict['functionname'] + '()'
        return sfunction

    else:
        sfunction = prefix + dict['functionname'] + '('

        try:
            for arg in dict.keys():
                if arg != 'functionname':
                    if type(dict[arg]) != str:
                        sfunction = sfunction + arg + '=' + str(dict[arg]) + ', '

                    else:
                        sfunction = sfunction + arg + '=' + "\'" + dict[arg] + "\'" + ', '

            sfunction = sfunction[:-2] + ')'

        except:
            print("Cannot implement the function, "+ dict['functionname'] +". Please double check." )
            raise ValueError('Please double check the transformations.')

        return sfunction






class UNet(pytorch_lightning.LightningModule):

    def __init__(self, inputs):
        super().__init__()


        self.userinputs = {
            # network parameters
            'dimensions': None,
            'in_channels': None,
            'out_channels':None,
            'channels':None,
            'strides': None,
            'kernel_size': 3,
            'up_kernel_size': 3,
            'num_res_units': 0,
            'act': 'PRELU',
            'norm': Norm.INSTANCE,
            'dropout': 0,
            # functions' parameters
            'lossfunction': None,
            'metricfunction': None,
            'roi_size': None,
            'sw_batch_size': None,
            'optimfunction':None
        }

        self.update_defaults(inputs)


        # TODO: Norm.GROUP cannot be called correctly-----------------------------------------------------------------------
        self._model = monai.networks.nets.UNet(
            dimensions=self.userinputs['dimensions'],
            in_channels=self.userinputs['in_channels'],
            out_channels=self.userinputs['out_channels'],
            channels=self.userinputs['channels'],
            strides=self.userinputs['strides'],
            kernel_size=self.userinputs['kernel_size'],
            up_kernel_size=self.userinputs['up_kernel_size'],
            num_res_units=self.userinputs['num_res_units'],
            act=self.userinputs['act'],
            dropout=self.userinputs['dropout'],
            norm=eval(self.userinputs['norm']))


        self.loss_function = eval(concatFunctionArguments(self.userinputs['lossfunction'], prefix='LOSS.'))
        self.val_metric = eval(concatFunctionArguments(self.userinputs['metricfunction'], prefix='METRIC.'))


        self.best_val_dice = None
        self.best_val_epoch = None


    def update_defaults(self, inputs):
        for input in inputs.keys():
            if input == 'norm' or input == 'normalization':
                self.userinputs['norm'] = 'Norm.' + inputs[input]
            else:
                self.userinputs[input] = inputs[input]



    def forward(self, x):
        return self._model(x)


    def prepare_data(self):
        # set deterministic training for reproducibility
        set_determinism(seed=0)


    def training_step(self, batch, batch_idx):
        images, labels = batch["IMAGE"], batch["SEGM"]
        output = self.forward(images)
        loss = self.loss_function(output, labels)
        return {"loss": loss}


    def training_epoch_end(self, outputs):
        # Only add the graph at the first epoch
        if self.current_epoch==1:
            sample_input = torch.rand((1,1)+self.userinputs['patch_size'])
            self.logger.experiment.add_graph(
                monai.networks.nets.UNet(
                    dimensions=self.userinputs['dimensions'],
                    in_channels=self.userinputs['in_channels'],
                    out_channels=self.userinputs['out_channels'],
                    channels=self.userinputs['channels'],
                    strides=self.userinputs['strides'],
                    num_res_units=self.userinputs['num_res_units'],
                    norm=eval(self.userinputs['norm'])
                ),
                [sample_input])

        # Calculate the average loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        # Logging at the end of every epoch
        self.logger.experiment.add_scalar('Loss/Train', avg_loss, self.current_epoch)
        return {'loss': avg_loss}


    def validation_step(self, batch, batch_idx):
        images, labels = batch["IMAGE"], batch["SEGM"]
        roi_size = self.userinputs['roi_size']
        sw_batch_size = self.userinputs['sw_batch_size']
        outputs = sliding_window_inference(images, roi_size, sw_batch_size, self.forward)

        loss = self.loss_function(outputs, labels)
        value = self.val_metric(y_pred=outputs, y=labels)

        # Save an image summary using MONAI
        if batch_idx==0:
            monai.visualize.img2tensorboard.plot_2d_or_3d_image(data=images,
                            step=self.current_epoch,
                            writer=self.logger.experiment,
                            tag='Input/Validation')
            monai.visualize.img2tensorboard.plot_2d_or_3d_image(data=outputs,
                            step=self.current_epoch,
                            writer=self.logger.experiment,
                            tag='Output/Validation')

        return {"val_loss": loss, "val_dice": value}



    def validation_epoch_end(self, outputs):
        # Calculate the average loss
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_dice = torch.stack([x['val_dice'] for x in outputs]).mean()
        # Logging at the end of every epoch
        self.logger.experiment.add_scalar('Loss/Val', avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar('Dice/Val', avg_dice, self.current_epoch)

        # Handle the first time
        if self.best_val_dice is None:
            self.best_val_dice = avg_dice
            self.best_val_epoch = self.current_epoch
        if avg_loss < self.best_val_dice:
            self.best_val_dice = avg_dice
            self.best_val_epoch = self.current_epoch
        return {"val_loss": avg_loss, "val_dice": avg_dice}


    def test_step(self, batch, batch_idx):
        images, labels = batch["IMAGE"], batch["SEGM"]
        roi_size = self.userinputs['roi_size']
        sw_batch_size = self.userinputs['sw_batch_size']
        outputs = sliding_window_inference(images, roi_size, sw_batch_size, self.forward)

        # TODO: conditions that no masks for testing image----------------------------------------
        value = self.val_metric(y_pred=outputs, y=labels)
        return {"test_dice": value}
        # TODO: conditions that no masks for testing image----------------------------------------



    def test_epoch_end(self, outputs):
        test_dice, num_items = 0, 0
        for output in outputs:
            test_dice += output["test_dice"].sum().item()
            num_items += len(output["test_dice"])
        mean_test_dice = torch.tensor(test_dice / num_items)
        tensorboard_logs = {
            "test_dice": mean_test_dice,
        }
        return {"log": tensorboard_logs}


    def configure_optimizers(self):
        sgfunc = concatFunctionArguments(self.userinputs['optimfunction'], prefix='OPTIMIZATION.')
        sgfunc = sgfunc[:sgfunc.index('(')+1] + 'self._model.parameters(), ' + sgfunc[sgfunc.index('(')+1:]
        return eval(sgfunc)
