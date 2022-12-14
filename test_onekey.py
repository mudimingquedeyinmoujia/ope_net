import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader
import datasets
import models
import utils
import myutils
from test import eval_both_ope, test_both_ope, single_img_sr


def main(exp_folder, test_config_list, test_ckpt):
    ckpt_name = test_ckpt.split('/')[-1].split('.')[-2]
    save_dir = os.path.join(exp_folder, 'ALL_TEST_folder/' + ckpt_name)
    os.makedirs(save_dir, exist_ok=True)
    log, _ = utils.set_save_path(save_dir, remove=False, writer=False)
    resume_path = os.path.join(exp_folder, test_ckpt)

    test_all_dic = {}
    if 'test_all_info.json' in os.listdir(save_dir):
        test_all_dic = myutils.load_json(path=os.path.join(save_dir, 'test_all_info.json'))
        tested_config_list = [key for key in test_all_dic.keys()]
        tested_config_list = sorted(tested_config_list)
    else:
        tested_config_list = []

    for test_config_file in test_config_list:
        test_config_name = test_config_file.split('/')[-1].split('.')[-2]
        if test_config_name in tested_config_list:
            pass
        else:
            log_name = test_config_name + '_log.txt'

            with open(os.path.join('configs/test-CIRnet', test_config_file), 'r') as f:
                test_config = yaml.load(f, Loader=yaml.FullLoader)

            test_spec = test_config['test_dataset']
            dataset = datasets.make(test_spec['dataset'])
            dataset = datasets.make(test_spec['wrapper'], args={'dataset': dataset})
            loader = DataLoader(dataset, batch_size=test_spec['batch_size'],
                                num_workers=8, pin_memory=True)

            sv_file = torch.load(resume_path, map_location=lambda storage, loc: storage)
            model = models.make(sv_file['model'], load_sd=True).cuda()

            test_psnr, test_ssim, test_run_time = test_both_ope(loader, model, log, log_name,
                                                                eval_type=test_config.get('eval_type'),
                                                                up_down=test_config.get('up_down'))

            log('test avg: psnr={:.4f}'.format(test_psnr), filename=log_name)
            log('test avg: ssim={:.4f}'.format(test_ssim), filename=log_name)
            log(f'test avg encoder time: {test_run_time[0]}s', filename=log_name)
            log(f'test avg render time: {test_run_time[1]}s', filename=log_name)
            log(f'test avg all time: {test_run_time[2]}s', filename=log_name)

            test_all_dic.update({test_config_name: [test_psnr, test_ssim]})
            myutils.save_json(path=os.path.join(save_dir, 'test_all_info.json'), save_dic=test_all_dic)


test_config_list = [
    # 'test_CIR-SR-div2k-x2-1.yaml',
    # 'test_CIR-SR-div2k-x2-2.yaml',
    # 'test_CIR-SR-b100-x2.yaml',
    # 'test_CIR-SR-b100-x3.yaml',
    # 'test_CIR-SR-b100-x4.yaml',
    # 'test_CIR-SR-b100-x6.yaml',
    # 'test_CIR-SR-b100-x8.yaml',
    # 'test_CIR-SR-div2k-x12.yaml',
    # 'test_CIR-SR-div2k-x18.yaml',
    # 'test_CIR-SR-div2k-x24.yaml',
    # 'test_CIR-SR-div2k-x3.yaml',
    # 'test_CIR-SR-div2k-x2.yaml',
    # 'test_CIR-SR-div2k-x30.yaml',
    # 'test_CIR-SR-div2k-x4.yaml',
    # 'test_CIR-SR-div2k-x6.yaml',
    'test_CIR-SR-set14-x2.yaml',
    'test_CIR-SR-set14-x3.yaml',
    'test_CIR-SR-set14-x4.yaml',
    'test_CIR-SR-set14-x6.yaml',
    'test_CIR-SR-set14-x8.yaml',
    # 'test_CIR-SR-set5-x2.yaml',
    # 'test_CIR-SR-set5-x3.yaml',
    # 'test_CIR-SR-set5-x4.yaml',
    # 'test_CIR-SR-set5-x6.yaml',
    # 'test_CIR-SR-set5-x8.yaml',
    # 'test_CIR-SR-urban100-x2.yaml',
    # 'test_CIR-SR-urban100-x3.yaml',
    # 'test_CIR-SR-urban100-x4.yaml',
    # 'test_CIR-SR-urban100-x6.yaml',
    # 'test_CIR-SR-urban100-x8.yaml',
    # up_down avg
    # 'up_down/avg/test_CIR-SR-set5-x2-avg2.yaml',
    # 'up_down/avg/test_CIR-SR-b100-x2-avg2.yaml',
    # 'up_down/avg/test_CIR-SR-b100-x2-avg3.yaml',
    # 'up_down/avg/test_CIR-SR-b100-x3-avg2.yaml',
    # 'up_down/avg/test_CIR-SR-b100-x3-avg3.yaml',
    # 'up_down/avg/test_CIR-SR-b100-x4-avg2.yaml',
    # 'up_down/avg/test_CIR-SR-b100-x4-avg3.yaml',
    # 'up_down/avg/test_CIR-SR-div2k-x2-avg2.yaml',
    # 'up_down/avg/test_CIR-SR-div2k-x2-avg3.yaml',
    # 'up_down/avg/test_CIR-SR-div2k-x3-avg2.yaml',
    # 'up_down/avg/test_CIR-SR-div2k-x3-avg3.yaml',
    # 'up_down/avg/test_CIR-SR-div2k-x4-avg2.yaml',
    # 'up_down/avg/test_CIR-SR-div2k-x4-avg3.yaml',
    # 'up_down/avg/test_CIR-SR-set14-x2-avg2.yaml',
    # 'up_down/avg/test_CIR-SR-set14-x2-avg3.yaml',
    # 'up_down/avg/test_CIR-SR-set14-x3-avg2.yaml',
    # 'up_down/avg/test_CIR-SR-set14-x3-avg3.yaml',
    # 'up_down/avg/test_CIR-SR-set14-x4-avg2.yaml',
    # 'up_down/avg/test_CIR-SR-set14-x4-avg3.yaml',
    # 'up_down/avg/test_CIR-SR-set5-x2-avg3.yaml',
    # 'up_down/avg/test_CIR-SR-set5-x3-avg2.yaml',
    # 'up_down/avg/test_CIR-SR-set5-x3-avg3.yaml',
    # 'up_down/avg/test_CIR-SR-set5-x4-avg2.yaml',
    # 'up_down/avg/test_CIR-SR-set5-x4-avg3.yaml',
    # 'up_down/avg/test_CIR-SR-urban100-x2-avg2.yaml',
    # 'up_down/avg/test_CIR-SR-urban100-x2-avg3.yaml',
    # 'up_down/avg/test_CIR-SR-urban100-x3-avg2.yaml',
    # 'up_down/avg/test_CIR-SR-urban100-x3-avg3.yaml',
    # 'up_down/avg/test_CIR-SR-urban100-x4-avg2.yaml',
    # 'up_down/avg/test_CIR-SR-urban100-x4-avg3.yaml',
    # up_down bic
    # 'up_down/bic/test_CIR-SR-set14-x2-ud3.yaml',
    # 'up_down/bic/test_CIR-SR-b100-x2-ud2.yaml',
    # 'up_down/bic/test_CIR-SR-b100-x2-ud3.yaml',
    # 'up_down/bic/test_CIR-SR-b100-x3-ud2.yaml',
    # 'up_down/bic/test_CIR-SR-b100-x3-ud3.yaml',
    # 'up_down/bic/test_CIR-SR-b100-x4-ud2.yaml',
    # 'up_down/bic/test_CIR-SR-b100-x4-ud3.yaml',
    # 'up_down/bic/test_CIR-SR-div2k-x2-ud2.yaml',
    # 'up_down/bic/test_CIR-SR-div2k-x2-ud3.yaml',
    # 'up_down/bic/test_CIR-SR-div2k-x3-ud2.yaml',
    # 'up_down/bic/test_CIR-SR-div2k-x3-ud3.yaml',
    # 'up_down/bic/test_CIR-SR-div2k-x4-ud2.yaml',
    # 'up_down/bic/test_CIR-SR-div2k-x4-ud3.yaml',
    # 'up_down/bic/test_CIR-SR-set14-x2-ud2.yaml',
    # 'up_down/bic/test_CIR-SR-set14-x3-ud2.yaml',
    # 'up_down/bic/test_CIR-SR-set14-x3-ud3.yaml',
    # 'up_down/bic/test_CIR-SR-set14-x4-ud2.yaml',
    # 'up_down/bic/test_CIR-SR-set14-x4-ud3.yaml',
    # 'up_down/bic/test_CIR-SR-set5-x2-ud2.yaml',
    # 'up_down/bic/test_CIR-SR-set5-x2-ud3.yaml',
    # 'up_down/bic/test_CIR-SR-set5-x3-ud2.yaml',
    # 'up_down/bic/test_CIR-SR-set5-x3-ud3.yaml',
    # 'up_down/bic/test_CIR-SR-set5-x4-ud2.yaml',
    # 'up_down/bic/test_CIR-SR-set5-x4-ud3.yaml',
    # 'up_down/bic/test_CIR-SR-urban100-x2-ud2.yaml',
    # 'up_down/bic/test_CIR-SR-urban100-x2-ud3.yaml',
    # 'up_down/bic/test_CIR-SR-urban100-x3-ud2.yaml',
    # 'up_down/bic/test_CIR-SR-urban100-x3-ud3.yaml',
    # 'up_down/bic/test_CIR-SR-urban100-x4-ud2.yaml',
    # 'up_down/bic/test_CIR-SR-urban100-x4-ud3.yaml',

]

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # exp_folder = 'save/_train_rdn-OPE-011_exp01'
    # test_ckpt = 'epoch-605.pth'
    # main(exp_folder=exp_folder, test_config_list=test_config_list, test_ckpt=test_ckpt)
    # print([f'up_down/bic/{i}' for i in os.listdir('configs/test-CIRnet/up_down/bic')])
    # batch
    exp_dict = {
        'save/_train_edsr-OPE-abl-aa_exp_01': ['epoch-1000.pth'],
        # 'save/_train_edsr-OPE-204_exp_02': ['epoch-300.pth'],
        # 'save/_train_edsr-OPE-abl-a-resume_exp01': ['epoch-990.pth', 'epoch-800.pth'],
        # 'save/_train_edsr-OPE-abl-b-resume_exp01': ['epoch-990.pth', 'epoch-510.pth'],
        # 'save/_train_edsr-OPE-abl-c-resume_exp01': ['epoch-990.pth'],
        # 'save/_train_rdn-OPE-015_exp01': ['epoch-780.pth'],
        # 'save/_train_edsr-OPE-abl-c-resume_exp01': ['epoch-600.pth', 'epoch-700.pth', 'epoch-800.pth',
        #                                             'epoch-1000.pth'],
        # 'save/_train_edsr-OPE-001-resume_exp01': ['epoch-990.pth'],
    }
    for key in exp_dict.keys():
        for test_ckpt in exp_dict[key]:
            main(exp_folder=key, test_config_list=test_config_list, test_ckpt=test_ckpt)
