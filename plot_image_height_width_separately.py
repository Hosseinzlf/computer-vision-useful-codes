# Plot pixels values for 2D image separately in height and width direction 
# Extract the pixel values along the height and width directions

height_values = np.mean(Masked_img, axis=1)  # Average pixel values along height
width_values = np.mean(Masked_img, axis=0)   # Average pixel values along width

plt.figure(figsize=(10, 5))

# Plot for height direction
plt.subplot(1, 2, 1)
plt.plot(range(Masked_img.shape[0]), height_values, c='b')
plt.xlabel('Height')
plt.ylabel('Pixel Value')
plt.title('Pixel Values along Height')

# Plot for width direction
plt.subplot(1, 2, 2)
plt.plot(range(Masked_img.shape[1]), width_values, c='b')
plt.xlabel('Width')
plt.ylabel('Pixel Value')
plt.title('Pixel Values along Width')

plt.tight_layout()
plt.show() 
