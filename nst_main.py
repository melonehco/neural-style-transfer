from neural_style_transfer import *

style_img = image_loader("images/style1.jpg")
content_img = image_loader("images/content1.JPG")

assert style_img.size() == content_img.size(), "input images must have same dimensions"

plt.ion()  # interactive mode on


plt.figure()
imshow(style_img, title="Style Image")
plt.figure()
imshow(content_img, title="Content Image")


cnn = models.vgg19(pretrained=True).features.to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)  # TODO: no idea where these numbers come from
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)


input_image = content_img.clone()
input_image.requires_grad = True
plt.figure()
imshow(input_image, 'input image')


output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_image)

plt.figure()
imshow(output, title='output image')
plt.ioff()  # interactive mode off
plt.show()
