import torch
import numpy as np
import matplotlib.pyplot as plt

from time import time


def trainModel(model, loss_fn, trainContent, valContent, trainStyle, valStyle, epochs=20, learningRate=0.001, weightDecay=1e-3, device="cpu", every=6):
  
    optimizer = torch.optim.Adam(model.parameters())#, lr=learningRate, weight_decay=weightDecay)
    
    min_valid_loss = np.inf
    epoch_losses = []
    val_losses = []

    for e in range(epochs):
        epoch_loss = 0.0
        val_loss = 0.0
        epoch_time = time()
        i=0.0

        model.train()
        for content, style in zip(trainContent, trainStyle):

            content, style = content[0].to(device), style[0].to(device)
            
            i+=1
            if i%every == 0:
              print('\rTime: {:.6f}s \tProgress: {:.3f}% \tLoss: {:.6f}'.format(
                      time()-epoch_time,i/len(trainContent)*100, epoch_loss), end="")
            
            optimizer.zero_grad()
            model.zero_grad()

            styledImage = model(content, style)
            
            content_in = model.target
            content_out = model.encoder(styledImage)
            styles_in = model.style_outputs[:4]
            styles_out = model.style_outputs[4:]

            loss = loss_fn(content_in, content_out, styles_in, styles_out)
            loss.backward()
            
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(trainContent) 
        epoch_losses.append(epoch_loss)      

        with torch.no_grad():
            model.eval()
            i = 0.0
            for content, style in zip(valContent, valStyle):

                content, style = content[0].to(device), style[0].to(device)

                i+=1
                if i%every == 0:
                  print('\rTime: {:.6f}s \tProgress: {:.3f}%'.format(
                      time()-epoch_time,i/len(valContent)*100), end="")

                styledImage = model(content, style)

                content_in = model.target
                content_out = model.encoder(styledImage)
                styles_in = model.style_outputs[:4]
                styles_out = model.style_outputs[4:]

                loss = loss_fn(content_in, content_out, styles_in, styles_out)

                val_loss += loss.item()
            
            val_loss /= len(valContent)
            val_losses.append(val_loss)
            
            print('\rEpoch: {} \tTime: {:.6f}s \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                    e, time()-epoch_time, epoch_loss, val_loss))
            
            if val_loss < min_valid_loss: 
                print(f"Found better model: Validation loss decreased {min_valid_loss} -> {val_loss}")
                min_valid_loss = val_loss
                torch.save(model.state_dict(), f'adain_trained_epoch_{e}')
    
    plt.figure(figsize=(20, 10))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.subplot(121)
    plt.plot(range(epochs), epoch_losses, color='r')
    plt.title("Training Loss")
    plt.subplot(122)
    plt.plot(range(epochs), val_losses, color='b')
    plt.title("Validation Loss")
    plt.show()
    torch.save(model.state_dict(), 'adain_trained')
    return model