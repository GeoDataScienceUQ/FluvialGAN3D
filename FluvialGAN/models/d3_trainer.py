from models.d3_model import D3Model
import torch
import os
def save_optimizer(optim, label, epoch, args):
    save_filename = '%s_optim_%s.pth' % (epoch, label)
    save_path = os.path.join(save_filename)
    torch.save(optim.state_dict(),save_path)

def load_optimizer(optim, label, epoch, args):
    save_filename = '%s_optim_%s.pth' % (epoch, label)
    save_path = os.path.join(save_filename)
    weights = torch.load(save_path)
    optim.load_state_dict(weights)
    return optim
class D3Trainer():
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """
    def __init__(self, args):
        self.args = args
        self.pix2pix_model = D3Model(args)
        if len(args.gpu_ids) > 0:
            self.pix2pix_model_on_one_gpu = self.pix2pix_model.cuda()

        self.generated = None
        self.isTrain = args.isTrain

        if args.isTrain:
            self.optimizer_G, self.optimizer_D = \
                self.pix2pix_model_on_one_gpu.create_optimizers(args)

            if args.is_continue:
                self.optimizer_G = load_optimizer(self.optimizer_G,'G', args.which_epoch, self.args)
                self.optimizer_D = load_optimizer(self.optimizer_D,'D', args.which_epoch, self.args)
                
    def test_generate(self, noise, data,num=1):
        generated = self.pix2pix_model(noise, data, mode='inference',num=num)
        return generated

    def run_generator_one_step(self, noise, data,num):
        self.optimizer_G.zero_grad()
        g_losses, generated = self.pix2pix_model(noise, data, mode='generator',num=num)
        g_loss = sum(g_losses.values()).mean()
        g_loss.backward()
        self.optimizer_G.step()
        self.g_losses = g_losses
        self.generated = generated

    def run_discriminator_one_step(self, noise, data,num):
        self.optimizer_D.zero_grad()
        d_losses = self.pix2pix_model(noise, data, mode='discriminator',num=num)
        d_loss = sum(d_losses.values()).mean()
        d_loss.backward()
        self.optimizer_D.step()
        self.d_losses = d_losses

    def get_latest_losses(self):
        return {**self.g_losses, **self.d_losses}

    def get_latest_generated(self):
        return self.generated

    def save(self, epoch):
        self.pix2pix_model_on_one_gpu.save(epoch)
        save_optimizer(self.optimizer_G,'G', epoch, self.args)
        save_optimizer(self.optimizer_D,'D', epoch, self.args)