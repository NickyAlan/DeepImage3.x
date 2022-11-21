import os 
import torch
import random
import shutil
import torchvision
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from timeit import default_timer as timer 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def ToJPG(image_path:str) :
    '''
    convert almost all image format(except .avif) to jpg format
    '''
    if image_path.endswith('avif') :
        print(f"can't convert the {image_path}\nbecause .avif format")
    if not image_path.endswith('jpg') and not image_path.endswith('avif') :
        image = Image.open(image_path).convert('RGB')
        image_name = image_path.split('.')[:-1]
        save_path = ''.join(image_name).strip('/')
        save_path = f'{save_path}.jpg'
        image.save(save_path)
        os.remove(image_path)


def CreateDataFolder(raw_data_dir = 'raw_data', dataset_dir = 'dataset',
                     train_size = 0.8, val_size = 0.15, test_size = 0.05, hasTest = True, shuffle=True) : 
    '''
    create dataset folder ; train -> class1, class2, class3 ...
                            val -> class1, class2, class3 ...
                            test -> class1, class2, class3 ...
    from raw_data folder ;  class1, class2, class3 ...
    '''

    for class_name in os.listdir(raw_data_dir) :
        class_path = os.path.join(raw_data_dir, class_name)
        for image_name in os.listdir(class_path) :
            image_path = os.path.join(class_path, image_name)
            ToJPG(image_path)
        
    if hasTest :
        assert train_size + val_size + test_size <= 1  , 'all of folder must be 100%'
        assert train_size > val_size and train_size >= test_size , 'train folder size have to more than val and test folder'
        splitFolder = ['train', 'val', 'test']
    else :
        assert train_size + val_size  == 1 , 'all of folder must be 100%'
        assert train_size > val_size , 'train folder size have to more than val folder'
        splitFolder = ['train', 'val']

    if not os.path.isdir(dataset_dir) :
        os.makedirs(dataset_dir)
        for folder in splitFolder :
            os.makedirs(os.path.join(dataset_dir, folder))

        for class_name in os.listdir(raw_data_dir) :
            main_folder = os.path.join(raw_data_dir, class_name)
            images_name = os.listdir(main_folder)
            total_image = len(images_name)
            
            if shuffle :
                random.shuffle(images_name)
            
            train_images = images_name[:int(train_size*total_image)]
            
            if hasTest :
                val_images = images_name[int(train_size*total_image):int(train_size*total_image)+int(val_size*total_image)]
                test_images = images_name[int(train_size*total_image)+int(val_size*total_image):]
            else :
                val_images = images_name[int(train_size*total_image):]
            
            for folder in splitFolder :
                os.makedirs(os.path.join(dataset_dir, folder, class_name))
            
            for image_name in train_images :
                source_path = os.path.join(raw_data_dir, class_name, image_name)
                desc_path = os.path.join(dataset_dir, 'train', class_name, image_name)
                shutil.copy(source_path, desc_path)
            
            for image_name in val_images :
                source_path = os.path.join(raw_data_dir, class_name, image_name)
                desc_path = os.path.join(dataset_dir, 'val', class_name, image_name)
                shutil.copy(source_path, desc_path)

            if hasTest :
                for image_name in test_images :
                    source_path = os.path.join(raw_data_dir, class_name, image_name)
                    desc_path = os.path.join(dataset_dir, 'test', class_name, image_name)
                    shutil.copy(source_path, desc_path)
        
        print(f'\tcreated {dataset_dir} succesfully.')
    else :
        print(f'already have {dataset_dir}.')


def ToTensor(image_path:str, image_size = (288, 288), mean=[0.485, 0.456, 0.406], std=[0.485, 0.456, 0.406]) :
    '''
    image_path to tensor for prediction
    '''
    transformer = transforms.Compose([ transforms.Resize(image_size), transforms.Normalize(mean, std) ])
    image = torchvision.io.read_image(image_path).type(torch.float32) / 255.
    image_transform = transformer(image)
    return image_transform.to(device)

def ToTensorShowImage(image_path:str, image_size = (300, 300)) :
    '''
    image_path to tensor for show when using prediction
    '''
    transformer = transforms.Resize(image_size)
    image = torchvision.io.read_image(image_path).type(torch.float32) / 255.
    image_transform = transformer(image)
    return image_transform.cpu()

def findMeanStd(loader:DataLoader) :
    '''
    find mean and std of dataloader
    '''

    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, _ in loader:
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2,
                                  dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)        
    return mean,std


def train_one_epoch(model, dataloader, loss_fn, optimizer) :
    model.train()
    train_acc, train_loss = 0, 0
    for batch, (images, labels) in enumerate(dataloader, start=1) :
         images, labels = images.to(device), labels.to(device)
         predict = model(images) 
         loss = loss_fn(predict, labels)
         train_loss += loss.item()

         optimizer.zero_grad()
         loss.backward()
         optimizer.step()

         predict_labels = torch.argmax(predict, dim=1) 
         train_acc += (predict_labels == labels).sum().item() / len(predict_labels)

         if batch ==1 or batch%5 == 0 or (batch == len(dataloader)) :
             print(f' {batch}/{len(dataloader)} batches')
    
    train_acc = train_acc / len(dataloader)
    train_loss = train_loss / len(dataloader) 

    return train_acc, train_loss


def test_one_epoch(model, dataloader, loss_fn) :
    model.eval()
    test_acc, test_loss = 0, 0
    with torch.inference_mode() :
        for batch, (images, labels) in enumerate(dataloader, start=1) :
            images, labels = images.to(device), labels.to(device)
            predict = model(images) 

            loss = loss_fn(predict, labels)
            test_loss += loss.item()

            predict_labels = torch.argmax(predict, dim=1)
            test_acc += (predict_labels == labels).sum().item() / len(predict_labels)

    test_acc = test_acc / len(dataloader)
    test_loss = test_loss / len(dataloader)

    return test_acc, test_loss

def train(
    model, train_dataloader, val_dataloader, loss_fn, optimizer, epochs, scheduler, model_name, class_names
) :
    result = {
        'train_acc' : [],
        'train_loss' : [],
        'val_acc' : [],
        'val_loss' : [],
    }
    start_time = timer()
    best_loss = 10e+4
    problemClassifier = 'MultiClassifier' if len(class_names) > 2 else 'BinaryClassifier'
    print(f'\n\tstart training {model_name}({problemClassifier})')
    for epoch in range(1, epochs+1) :
        print(f'epochs {epoch}/{epochs}')
        train_acc, train_loss = train_one_epoch(model, train_dataloader, loss_fn, optimizer)
        val_acc, val_loss = test_one_epoch(model, val_dataloader, loss_fn)
        scheduler.step(val_loss)
    
        if val_loss < best_loss :
            best_loss = val_loss 
            save_path = f'best_{model_name}_weights.pt'
            torch.save(model.state_dict(), save_path)
            print(f'\n\tsave {model_name} with validation loss = {val_loss:.6f}')
            
            best_epoch = f'best epoch {epoch} val_acc {val_acc*100:.2f}%  val_loss {val_loss:.4f}'
        
        print(f'''
        \n\t[SUMMARY] epoch {epoch}/{epochs}
        \ttrain_acc {train_acc*100:.2f}% | val_acc {val_acc*100:.2f}%
        \ttrain_loss {train_loss:.4f} | val_loss {val_loss:.4f}
        ''')

        result['train_acc'].append(train_acc)
        result['train_loss'].append(train_loss)
        result['val_acc'].append(val_acc)
        result['val_loss'].append(val_loss)
    
    end_time = timer()
    total_time = (end_time-start_time) / 60
    print(f"Total training time: {total_time:.3f} minutes")
    print(f'[INFO] {best_epoch}')

    return result

def predict(image_tensor:torch.tensor, model, class_names:list) :
    model.eval()
    with torch.inference_mode() :
        predict = model(image_tensor.unsqueeze(dim=0))
        probas = torch.softmax(predict, dim=1) 
        predict_idx  = torch.argmax(probas, dim=1) 
        
        predict_class = class_names[predict_idx]
    return predict_class, probas[0].detach().cpu().numpy()

def plot_predict(image_tensor:torch.tensor, model, show_image, class_names:list, color, add_title=None) :
    class_, probas = predict(image_tensor, model, class_names)
    confident = np.max(probas)*100
    plt.imshow(show_image.permute(1,2,0))
    title_ = f'{class_} {confident:.2f}%' 
    if add_title :
        title_ += f' / {add_title}' 
    plt.title(title_, color=color)
    plt.axis('off')

def plot_predict_most_confs(image_path:str, most_like:int, class_names, model, figsize=(13,5)) :
    most_like = most_like if len(class_names) > most_like else len(class_names)
    image_tensor = ToTensor(image_path)
    pred_class, confs = predict(image_tensor, model, class_names)
    most_confs_idx = np.argsort(confs)[::-1][:most_like]
    most_confs = confs[most_confs_idx]
    most_confs_class = np.array(class_names)[most_confs_idx]

    plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    plt.title(f'{pred_class} {most_confs[0]*100:.2f}%')
    plt.imshow(ToTensorShowImage(image_path).permute(1,2,0))
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.barh(np.arange(most_like)[::-1], most_confs*100, color='orange', edgecolor='violet')
    plt.yticks(np.arange(most_like)[::-1], labels=most_confs_class)
    plt.xlabel('confident(%)');