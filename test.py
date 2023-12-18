import torch
import os

work_dir = '/home/czy/Documents/H-InDex'

def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.items(), model_2.items()):
        # key_item_1[1] = key_item_1[1].to('cpu')
        # key_item_2[1] = key_item_2[1].to('cpu')
        # print("Model 1:", key_item_1[0], key_item_1[1].shape)
        if torch.equal(key_item_1[1].to('cuda'), key_item_2[1].to('cuda')):
            print('Models match for', key_item_1[0])
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                print("Name and value different at ", key_item_1[0], " and ", key_item_2[0])
                # raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')

def test_params():
    # NOTE: the two model has different keys, so we can't directly use this 
    # Test frankmocap and adapted model data
    frankmocap_model = torch.load(os.path.join(work_dir, 'archive/frankmocap_hand.pth'))
    hammering_model = torch.load(os.path.join(work_dir, 'archive/adapted_frankmocap_hand_ckpts/hammer-v0.pth'))

    # hammering_model = hammering_model.to('cpu')
    # frankmocap_model = frankmocap_model.to('cpu')

    compare_models(hammering_model, frankmocap_model)


def main():
    test_params()

if __name__ == '__main__':
    main()