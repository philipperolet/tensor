# coding: utf-8

import os
import torch
import torchvision as tv
from PIL import Image
from practical4 import CustomNet, default_trainer, args


def test_model_parameters_saving():
    torch.manual_seed(0)
    test_error = default_trainer.train()
    model_data = default_trainer.model.state_dict()
    torch.save(model_data, "model_data.txt")
    new_model = CustomNet()
    new_model.load_state_dict(torch.load("model_data.txt"))
    default_trainer.model = new_model
    new_model_error = default_trainer._compute_test_error()
    print(test_error, new_model_error)
    assert test_error == new_model_error, "{} != {}".format(test_error, new_model_error)


def get_model(filename):
    torch.manual_seed(0)

    if not(os.path.isfile(filename)):
        test_error = default_trainer.train()
        model_data = default_trainer.model.state_dict()
        torch.save(model_data, filename)
    else:
        new_model = CustomNet()
        new_model.load_state_dict(torch.load(filename))
        default_trainer.model = new_model
        test_error = default_trainer._compute_test_error()

    return test_error, default_trainer.model


if __name__ == '__main__':
    test_error, model = get_model(args.modelfile)
    print("Example MNIST digit", default_trainer.data['training_input'][0])
    print("Test error: {}%".format(test_error * 100))

    for i in [1, 9, 8, 4, 3, 5]:
        image = Image.open(os.path.join(args.imagepath, "{}.jpg".format(i)))
        
        tensor = tv.transforms.ToTensor()(image)
        tensor = tv.transforms.Normalize([tensor.mean()], [-tensor.std()])(tensor)
        tensor = tensor.apply_(lambda x: -0.5 if x < 0 else x)
        tensor = torch.unsqueeze(tensor, 0)
        with torch.no_grad():
            prediction = torch.argmax(model(tensor))
            print(model(tensor))
            print("Prediction for {} : {}".format(i, prediction))
