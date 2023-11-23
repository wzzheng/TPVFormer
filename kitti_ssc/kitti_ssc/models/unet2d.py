"""
Code adapted from https://github.com/shariqfarooq123/AdaBins/blob/main/models/unet_adaptive_bins.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class UpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSampleBN, self).__init__()
        self._net = nn.Sequential(
            nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_features),
            nn.LeakyReLU(),
            nn.Conv2d(
                output_features, output_features, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(output_features),
            nn.LeakyReLU(),
        )

    def forward(self, x, concat_with):
        up_x = F.interpolate(
            x,
            size=(concat_with.shape[2], concat_with.shape[3]),
            mode="bilinear",
            align_corners=True,
        )
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)


class DecoderBN(nn.Module):
    def __init__(
        self, num_features, bottleneck_features, out_feature, use_decoder=True
    ):
        super(DecoderBN, self).__init__()
        features = int(num_features)
        self.use_decoder = use_decoder

        self.conv2 = nn.Conv2d(
            bottleneck_features, features, kernel_size=1, stride=1, padding=1
        )

        self.out_feature_1_1 = out_feature
        self.out_feature_1_2 = out_feature
        self.out_feature_1_4 = out_feature
        self.out_feature_1_8 = out_feature
        self.out_feature_1_16 = out_feature
        self.feature_1_16 = features // 2
        self.feature_1_8 = features // 4
        self.feature_1_4 = features // 8
        self.feature_1_2 = features // 16
        self.feature_1_1 = features // 32

        if self.use_decoder:
            self.resize_output_1_1 = nn.Conv2d(
                self.feature_1_1, self.out_feature_1_1, kernel_size=1
            )
            self.resize_output_1_2 = nn.Conv2d(
                self.feature_1_2, self.out_feature_1_2, kernel_size=1
            )
            self.resize_output_1_4 = nn.Conv2d(
                self.feature_1_4, self.out_feature_1_4, kernel_size=1
            )
            self.resize_output_1_8 = nn.Conv2d(
                self.feature_1_8, self.out_feature_1_8, kernel_size=1
            )
            # self.resize_output_1_16 = nn.Conv2d(
            #     self.feature_1_16, self.out_feature_1_16, kernel_size=1
            # )

            self.up16 = UpSampleBN(
                skip_input=features + 224, output_features=self.feature_1_16
            )
            self.up8 = UpSampleBN(
                skip_input=self.feature_1_16 + 80, output_features=self.feature_1_8
            )
            self.up4 = UpSampleBN(
                skip_input=self.feature_1_8 + 48, output_features=self.feature_1_4
            )
            self.up2 = UpSampleBN(
                skip_input=self.feature_1_4 + 32, output_features=self.feature_1_2
            )
            self.up1 = UpSampleBN(
                skip_input=self.feature_1_2 + 3, output_features=self.feature_1_1
            )
        else:
            self.resize_output_1_1 = nn.Conv2d(3, out_feature, kernel_size=1)
            self.resize_output_1_2 = nn.Conv2d(32, out_feature * 2, kernel_size=1)
            self.resize_output_1_4 = nn.Conv2d(48, out_feature * 4, kernel_size=1)

    def forward(self, features):
        # x_block0, x_block1, x_block2, x_block3, x_block4 = (
        #     features[4],
        #     features[5],
        #     features[6],
        #     features[8],
        #     features[11],
        # )
        x_block0, x_block1, x_block2, x_block3, x_block4 = (
            features[1],
            features[2],
            features[3],
            features[4],
            features[5],
        )
        bs = x_block0.shape[0]
        x_d0 = self.conv2(x_block4)

        if self.use_decoder:
            x_1_16 = self.up16(x_d0, x_block3)
            x_1_8 = self.up8(x_1_16, x_block2)
            x_1_4 = self.up4(x_1_8, x_block1)
            x_1_2 = self.up2(x_1_4, x_block0)
            x_1_1 = self.up1(x_1_2, features[0])
            # return {
            #     "1_1": self.resize_output_1_1(x_1_1),
            #     "1_2": self.resize_output_1_2(x_1_2),
            #     "1_4": self.resize_output_1_4(x_1_4),
            #     "1_8": self.resize_output_1_8(x_1_8),
            #     "1_16": self.resize_output_1_16(x_1_16),
            # }
            return self.resize_output_1_1(x_1_1), self.resize_output_1_2(x_1_2), \
                self.resize_output_1_4(x_1_4), self.resize_output_1_8(x_1_8)

        else:
            x_1_1 = features[0]
            x_1_2, x_1_4, x_1_8, x_1_16 = (
                features[4],
                features[5],
                features[6],
                features[8],
            )
            x_global = features[-1].reshape(bs, 2560, -1).mean(2)
            return {
                "1_1": self.resize_output_1_1(x_1_1),
                "1_2": self.resize_output_1_2(x_1_2),
                "1_4": self.resize_output_1_4(x_1_4),
                "global": x_global,
            }


class Encoder(nn.Module):
    def __init__(self, backend):
        super(Encoder, self).__init__()
        self.original_model = backend

    def forward(self, x):
        features = [x]
        for k, v in self.original_model._modules.items():
            if k == "blocks":
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))
        # return features
        return features[0], features[4], features[5], features[6], features[8], features[11]


class UNet2D(nn.Module):
    def __init__(
        self, backend, num_features, out_feature, use_decoder=True,
        decoder_checkpoint=False,
    ):
        super(UNet2D, self).__init__()
        self.use_decoder = use_decoder
        self.encoder = Encoder(backend)
        self.decoder = DecoderBN(
            out_feature=out_feature,
            use_decoder=use_decoder,
            bottleneck_features=num_features,
            num_features=num_features,
        )
        self.decoder_checkpoint = decoder_checkpoint

    def forward(self, x, **kwargs):
        encoded_feats = self.encoder(x)
        if self.decoder_checkpoint:
            # unet_out = self.decoder(encoded_feats, **kwargs)
            unet_out = torch.utils.checkpoint.checkpoint(self.decoder, encoded_feats)
        else:
            unet_out = self.decoder(encoded_feats, **kwargs)
        return unet_out

    def get_encoder_params(self):  # lr/10 learning rate
        return self.encoder.parameters()

    def get_decoder_params(self):  # lr learning rate
        return self.decoder.parameters()

    @classmethod
    def build(cls, **kwargs):
        basemodel_name = "tf_efficientnet_b7_ns"
        num_features = 2560

        print("Loading base model ()...".format(basemodel_name), end="")
        basemodel = torch.hub.load(
            "rwightman/gen-efficientnet-pytorch", basemodel_name, pretrained=True
        )
        print("Done.")

        # Remove last layer
        print("Removing last two layers (global_pool & classifier).")
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()
        basemodel.bn2 = nn.Identity()
        basemodel.act2 = nn.Identity()

        # Building Encoder-Decoder model
        print("Building Encoder-Decoder model..", end="")
        m = cls(basemodel, num_features=num_features, **kwargs)
        print("Done.")
        return m

if __name__ == '__main__':
    model = UNet2D.build(out_feature=256, use_decoder=True)
