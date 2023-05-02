
def create_model(opt):
    model = None
    print(opt.model)

    if opt.model == 'resnet18' or opt.model == 'resnet34' or opt.model == 'resnet50' or opt.model == 'resnet101':
        from .posenet_model import PoseNetModel
        model = PoseNetModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (opt.model))
    return model

