import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG16_Weights


class FCNSkipNerveDefinitorPrunnable(nn.Module):
    def __init__(self,
                 num_classes=1,
                 input_channels=3,
                 base=64,
                 # dropout_probability=0.3,
                 # use_dropout=False,
                 # use_quantize=False,
                 use_inplace=True):
        super(FCNSkipNerveDefinitorPrunnable, self).__init__()

        dropout_probability=0.0
        use_dropout=False

        # self.use_quantize = use_quantize

        # if use_dropout:
        #     self.step = 3
        # else:
        #     self.step = 2

        # if use_quantize:
        #     self.quant = torch.quantization.QuantStub()
        #     self.dequant = torch.quantization.DeQuantStub()

        self.encoder1 = self._conv_block_2(input_channels, base,
                                           dropout_probability=dropout_probability,
                                           use_dropout=use_dropout,
                                           inplace=use_inplace)
        self.encoder2 = self._conv_block_2(base, base * 2,
                                           dropout_probability=dropout_probability,
                                           use_dropout=use_dropout,
                                           inplace=use_inplace)
        self.encoder3 = self._conv_block_3(base * 2, base * 4,
                                           dropout_probability=dropout_probability,
                                           use_dropout=use_dropout,
                                           inplace=use_inplace)
        self.encoder4 = self._conv_block_3(base * 4, base * 8,
                                           dropout_probability=dropout_probability,
                                           use_dropout=use_dropout,
                                           inplace=use_inplace)

        if use_dropout:
            self.intermidiate = nn.Sequential(
                nn.Conv2d(base * 8, base * 8, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=use_inplace),
                nn.Dropout(p=dropout_probability)
            )
        else:
            self.intermidiate = nn.Sequential(
                nn.Conv2d(base * 8, base * 8, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=use_inplace)
            )

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Decoder
        self.decoder4 = self._upconv_block(base * 8 + base * 4, base * 4,
                                           dropout_probability=dropout_probability,
                                           use_dropout=use_dropout,
                                           inplace=use_inplace)
        self.decoder3 = self._upconv_block(base * 4 + base * 2, base * 2,
                                           dropout_probability=dropout_probability,
                                           use_dropout=use_dropout,
                                           inplace=use_inplace)
        self.decoder2 = self._upconv_block(base * 2 + base, base,
                                           dropout_probability=dropout_probability,
                                           use_dropout=use_dropout,
                                           inplace=use_inplace)

        self.decoder_1_final_conv = nn.Sequential(
            nn.Conv2d(base, num_classes, kernel_size=1),
            nn.Sigmoid()
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)



    @staticmethod
    def _conv_block_2(in_channels, out_channels,
                 dropout_probability=0.5,
                 use_dropout=False,
                 inplace=True):
        if use_dropout:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=inplace),
                nn.Dropout(p=dropout_probability),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=inplace),
                nn.Dropout(p=dropout_probability)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=inplace),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=inplace)
            )

    @staticmethod
    def _conv_block_3(in_channels, out_channels,
                 dropout_probability=0.5,
                 use_dropout=False,
                 inplace=True):
        if use_dropout:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=inplace),
                nn.Dropout(p=dropout_probability),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=inplace),
                nn.Dropout(p=dropout_probability),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=inplace),
                nn.Dropout(p=dropout_probability)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=inplace),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=inplace),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=inplace)
            )

    @staticmethod
    def _upconv_block(in_channels, out_channels,
                 dropout_probability=0.5,
                 use_dropout=False,
                 inplace=True):
        if use_dropout:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=inplace),
                nn.Dropout(p=dropout_probability),
                nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=inplace),
                nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
            )

    def forward(self, x):

        # if self.use_quantize:
        #     x = self.quant(x)

        enc1 = self.encoder1(x)
        pool1 = self.pool(enc1)

        enc2 = self.encoder2(pool1)
        pool2 = self.pool(enc2)

        enc3 = self.encoder3(pool2)
        pool3 = self.pool(enc3)

        enc4 = self.encoder4(pool3)
        pool4 = self.pool(enc4)

        interm = self.intermidiate(pool4)
        # Decoder
        up4 = self.upsample(interm)

        dec4 = self.decoder4(torch.cat([up4, pool3], dim=1))
        dec3 = self.decoder3(torch.cat([dec4, pool2], dim=1))
        dec2 = self.decoder2(torch.cat([dec3, pool1], dim=1))

        ret = self.decoder_1_final_conv(dec2)

        # if self.use_quantize:
        #     ret = self.dequant(ret)

        return ret
    #
    # def prune_conv_block_2(self, encoder, pruning_ratio=0.5):
    #     i = 0
    #     remap_mask = prune_layer_weights(encoder[i], pruning_ratio)
    #     adjust_next_layer_input_weights(encoder[i + self.step], remap_mask)
    #     i += self.step
    #     remap_mask = prune_layer_weights(encoder[i], pruning_ratio)
    #
    #     return remap_mask
    #
    # def prune_conv_block_3(self, encoder, pruning_ratio=0.5):
    #     i = 0
    #     remap_mask = prune_layer_weights(encoder[i], pruning_ratio)
    #     adjust_next_layer_input_weights(encoder[i + self.step], remap_mask)
    #     i += self.step
    #     remap_mask = prune_layer_weights(encoder[i], pruning_ratio)
    #     adjust_next_layer_input_weights(encoder[i + self.step], remap_mask)
    #
    #     i += self.step
    #     remap_mask = prune_layer_weights(encoder[i], pruning_ratio)
    #
    #     return remap_mask
    #
    # def prune_upconv_block(self, encoder, pruning_ratio=0.5):
    #     i = 0
    #     remap_mask = prune_layer_weights(encoder[i], pruning_ratio)
    #     adjust_next_layer_input_weights(encoder[i + self.step], remap_mask)
    #     i += self.step
    #     remap_mask = prune_layer_weights(encoder[i], pruning_ratio)
    #
    #     return remap_mask

    def add_dropout(self, dropout_prob=0.3):
        def dropout_block(seq_model_orig):
            new_layers = []
            for layer in seq_model_orig.children():
                new_layers.append(layer)  # Add existing layer
                if isinstance(layer, nn.LeakyReLU):  # Add dropout after every ReLU
                    new_layers.append(nn.Dropout(p=dropout_prob))

            return nn.Sequential(*new_layers)

        self.step = 3

        self.encoder1 = dropout_block(self.encoder1)
        self.encoder2 = dropout_block(self.encoder2)
        self.encoder3 = dropout_block(self.encoder3)
        self.encoder4 = dropout_block(self.encoder4)

        self.intermidiate = dropout_block(self.intermidiate)

        self.decoder4 = dropout_block(self.decoder4)
        self.decoder3 = dropout_block(self.decoder3)
        self.decoder2 = dropout_block(self.decoder2)

    # def prune_layers(self, pruning_ratio=0.5):
    #
    #     enc1_mask = self.prune_conv_block_2(self.encoder1, pruning_ratio)
    #     adjust_next_layer_input_weights(self.encoder2[0], enc1_mask)
    #
    #     enc2_mask = self.prune_conv_block_2(self.encoder2, pruning_ratio)
    #     adjust_next_layer_input_weights(self.encoder3[0], enc2_mask)
    #
    #     enc3_mask = self.prune_conv_block_3(self.encoder3, pruning_ratio)
    #     adjust_next_layer_input_weights(self.encoder4[0], enc3_mask)
    #
    #     enc4_mask = self.prune_conv_block_3(self.encoder4, pruning_ratio)
    #     adjust_next_layer_input_weights(self.intermidiate[0], enc4_mask)
    #
    #     intermidiate_mask = prune_layer_weights(self.intermidiate[0], pruning_ratio)
    #     adjust_next_layer_input_weights(self.decoder4[0], torch.cat([intermidiate_mask, enc3_mask]))
    #
    #     dec4_mask = self.prune_upconv_block(self.decoder4, pruning_ratio)
    #     adjust_next_layer_input_weights(self.decoder3[0], torch.cat([dec4_mask, enc2_mask]))
    #
    #     dec3_mask = self.prune_upconv_block(self.decoder3, pruning_ratio)
    #     adjust_next_layer_input_weights(self.decoder2[0], torch.cat([dec3_mask, enc1_mask]))
    #
    #     dec2_mask = self.prune_upconv_block(self.decoder2, pruning_ratio)
    #
    #     prune_only_input_weights(self.decoder_1_final_conv[0], dec2_mask)

    # this is used in conjunction with quantization changes from outside the model
    # def add_quantize(self):
    #     self.use_quantize = True
    #     self.quant = torch.quantization.QuantStub()
    #     self.dequant = torch.quantization.DeQuantStub()

    # @staticmethod
    # def create_model(num_classes=1,
    #                  input_channels=3,
    #                  use_dropout=False,
    #                  dropout_probability=0.3,
    #                  in_place=True):
    #     # Load VGG-16 pretrained model
    #     vgg16 = models.vgg16(weights=VGG16_Weights.DEFAULT)
    #     vgg_features = list(vgg16.features.children())  # Extract VGG-16 feature blocks
    #
    #     # Create the custom model
    #     model = FCNSkipNerveDefinitorPrunnable(num_classes=num_classes,
    #                                            input_channels=input_channels,
    #                                            use_inplace=in_place)
    #
    #     # Map pretrained weights from VGG-16 to the custom model
    #     vgg_layers = [
    #         model.encoder1,  # Map first two VGG-16 blocks to encoder1
    #         model.encoder2,  # Map next two VGG-16 blocks to encoder2
    #         model.encoder3,  # Map next three VGG-16 blocks to encoder3
    #         model.encoder4  # Map next three VGG-16 blocks to encoder4
    #     ]
    #
    #     start_idx = 0
    #     for i, encoder in enumerate(vgg_layers):
    #         num_layers = len(list(encoder.children()))
    #         for j in range(num_layers):
    #             if isinstance(encoder[j], nn.Conv2d):
    #                 encoder[j].weight.data = vgg_features[start_idx].weight.data.clone()
    #                 encoder[j].bias.data = vgg_features[start_idx].bias.data.clone()
    #             start_idx += 1
    #         # include pooling - it is not included in encoder layer for skip connections
    #         start_idx += 1
    #
    #     if use_dropout:
    #         model.add_dropout(dropout_prob=dropout_probability)
    #
    #     # Return the fully initialized model
    #     return model


class HandmadeGlaucomaClassifier(nn.Module):
    def __init__(self,
                 num_classes=3,
                 input_channels=3,
                 base=64,
                 input_size=128):
        super(HandmadeGlaucomaClassifier, self).__init__()

        # Transfer weights from VGG16 for the first 4 blocks
        self.encoder1 = self._conv_block_2(input_channels, base)
        self.encoder2 = self._conv_block_2(base, base * 2)
        self.encoder3 = self._conv_block_3(base * 2, base * 4)
        self.encoder4 = self._conv_block_3(base * 4, base * 8)

        # Intermediate layer
        self.intermidiate = nn.Sequential(
            nn.Conv2d(base * 8, base * 8, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
        )

        # Fully connected layers for 3 output channels
        self.flatten = nn.Flatten()

        lin1 = nn.Linear(base * 8 * int((input_size / 16) ** 2), base)
        lin2 = nn.Linear(base, num_classes)

        self.fc = nn.Sequential(
            lin1,  # nn.Linear(base*8 * 7 * 7, 256),  # Adjust based on output of encoder4
            nn.ReLU(),
            lin2,  # nn.Linear(256, num_classes),
            nn.Sigmoid()
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    @staticmethod
    def _conv_block_2(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def _conv_block_3(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def _extract_vgg_weights(vgg, start, end):
        # Extracts a subset of VGG layers' weights as a state_dict
        sublayers = list(vgg.features.children())[start:end]
        state_dict = {f"{i}": layer.state_dict() for i, layer in enumerate(sublayers)}
        return {k: v for d in state_dict.values() for k, v in d.items()}

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        pool1 = self.pool(enc1)

        enc2 = self.encoder2(pool1)
        pool2 = self.pool(enc2)

        enc3 = self.encoder3(pool2)
        pool3 = self.pool(enc3)

        enc4 = self.encoder4(pool3)
        pool4 = self.pool(enc4)

        # Intermediate processing
        interm = self.intermidiate(pool4)

        # Flatten and FC output
        flattened = self.flatten(interm)
        output = self.fc(flattened)

        return output
