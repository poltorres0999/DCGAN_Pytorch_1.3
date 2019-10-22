import argparse
import configparser

from src.dcgan.dcgan import *
from src.dcgan.discriminator import Discriminator
from src.dcgan.generator import Generator
from src.dcgan.utils import *
from datetime import datetime

DEFAULT_CONFIG_PATH = "../../res/default_config.ini"


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cp', '--config_path', help="Path to configuration file '.ini.")
    parser.add_argument('-lm', '--load_models', action='store_true',
                        help="If active, loads the specified models (config.ini)")
    parser.add_argument('-d', '--enable_debug', action='store_true', help="Enable debug option")
    parser.add_argument('-plt', '--enable_plot', action='store_true', help="Plots will be shown during execution")
    parser.add_argument('-bds', '--base_data_set', help="The net will be trained with the specified "
                                                        "ptyroch base dataset. ex: MNIST")

    return parser.parse_args()


def process_args(args):
    config_path = DEFAULT_CONFIG_PATH
    load_models = False
    debug_enabled = False
    plot_enabled = False
    base_dataset = None

    if args.config_path:
        config_path = args.config_path
    if args.load_models:
        load_models = True
    if args.enable_debug is not None:
        debug_enabled = args.enable_debug
    if args.enable_plot is not None:
        plot_enabled = args.enable_plot
    if args.base_data_set is not None:
        base_dataset = args.base_data_set

    return config_path, load_models, debug_enabled, plot_enabled, base_dataset


def load_config_file(file_path):
    try:
        conf = configparser.ConfigParser()
        conf.read(file_path)
        return conf
    except configparser.Error as e:
        raise Exception(f"Error while parsing configuration file {file_path}, error: {e}")


def main():
    # Read command line args
    config_path, load_models, debug_enabled, plot_enabled, base_dataset = process_args(load_args())
    # Load config file
    conf = load_config_file(config_path)

    # Environment conf
    seed_min = int(conf['environment']['seed_min'])
    seed_max = int(conf['environment']['seed_max'])
    n_gpu = int(conf['environment']['n_gpu'])
    # Data conf
    batch_size = int(conf['data']['batch_size'])
    dl_workers = int(conf['data']['dl_workers'])
    data_root = conf['data']['data_root']
    data_set_path = conf['data']['data_set_path']
    image_size = int(conf['data']['image_size'])
    img_channels = int(conf['data']['img_channels'])
    norm = float(conf['data']['norm'])
    shuffle = True  # Todo parse to boolean
    # Training conf
    lr = float(conf['training']['lr'])
    num_epochs = int(conf['training']['num_epochs'])
    beta1 = float(conf['training']['beta1'])
    gen_temp_save_path = conf['training']['gen_save_path']
    disc_temp_save_path = conf['training']['disc_save_path']
    # Network conf
    noise_vector_size = int(conf['net']['noise_vector_size'])
    g_fm_depth = int(conf['net']['g_fm_depth'])
    d_fm_depth = int(conf['net']['d_fm_depth'])
    # results
    results_path = conf['results']['results_path']
    images_path = conf['results']['images_path']
    gen_model_path = conf['results']['gen_model_path']
    disc_model_path = conf['results']['disc_model_path']


    dataset_name = ""
    # Sets the device (GPU or CPU) and creates the data_loader
    device = set_environment((seed_min, seed_max), n_gpu)
    # Set image transformations
    transform = set_transform(img_size=image_size, img_depth=img_channels, norm=norm)
    if base_dataset is not None:
        data_loader = load_base_dataset(data_root=data_root, transform=transform, batch_size=batch_size,
                                        norm=norm, shuffle=shuffle, data_loader_workers=dl_workers,
                                        dataset_name=base_dataset)
        dataset_name = base_dataset
    else:
        data_loader = load_image_data(data_root=data_set_path, transform=transform, batch_size=batch_size,
                                      norm=norm, shuffle=shuffle, data_loader_workers=dl_workers)
        dataset_name = data_set_path.split("/")[-1]

    # Create or load Generator and Discriminator models
    if load_models:
        generator_net = Generator(n_gpu, noise_vector_size, fm_depth=g_fm_depth, img_nc=img_channels)
        gen_optimizer = optim.Adam(generator_net.parameters(), lr=lr, betas=(beta1, 0.999))
        generator_net, gen_optimizer, num_epochs = load_model(generator_net, gen_optimizer, gen_model_path)

        discriminator_net = Discriminator(n_gpu=n_gpu, fm_depth=d_fm_depth, img_nc=img_channels)
        disc_optimizer = optim.Adam(discriminator_net.parameters(), lr=lr, betas=(beta1, 0.999))
        discriminator_net, disc_optimizer, num_epochs = load_model(discriminator_net, disc_optimizer, disc_model_path)

    else:
        generator_net = Generator(n_gpu, noise_vector_size, fm_depth=g_fm_depth, img_nc=img_channels)
        gen_optimizer = optim.Adam(generator_net.parameters(), lr=lr, betas=(beta1, 0.999))
        discriminator_net = Discriminator(n_gpu=n_gpu, fm_depth=d_fm_depth, img_nc=img_channels)
        disc_optimizer = optim.Adam(discriminator_net.parameters(), lr=lr, betas=(beta1, 0.999))

    # Create loss function (Just BCE for now)
    loss_fn = nn.BCELoss()
    # Create DCGAN
    dc_gan = DCGAN(generator=generator_net, discriminator=discriminator_net, data_loader=data_loader, device=device)
    print(dc_gan)
    # Set loss function(BCE) and both model optimizers
    dc_gan.set_loss_fn(loss_fn)
    dc_gan.set_optimizers(disc_optimizer, gen_optimizer)
    # Start training loop
    start_date = datetime.now()
    dc_gan.train(num_epochs=num_epochs, noise_v_size=noise_vector_size,
                 disc_save_path=disc_temp_save_path, gen_save_path=gen_temp_save_path)
    # Plot training results
    plot_loss_resutls(dc_gan.gen_loss, dc_gan.disc_loss)
    save_loss_plot(dc_gan.gen_loss, dc_gan.disc_loss, results_path)
    # Save models
    torch.save(dc_gan.generator_net.state_dict(), gen_model_path)
    torch.save(dc_gan.discriminator_net.state_dict(), disc_model_path)
    # Create report
    create_report(r_path=results_path, s_date=start_date, gen_net=dc_gan.generator_net,
                  disc_net=dc_gan.discriminator_net, optimizer="Adam", loss_fn=dc_gan.loss_fn,
                  dataset_name=dataset_name, lr=lr, epochs=num_epochs, disc_loss=dc_gan.disc_loss,
                  gen_loss=dc_gan.gen_loss)

    # todo: generate and store images

if __name__ == "__main__":
    main()
