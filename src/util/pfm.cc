#include "util/pfm.h"

namespace colmap {

cv::Mat LoadPFM(const std::string& file_path) {
  // Open binary file
  std::ifstream file(file_path.c_str(), std::ios::in | std::ios::binary);

  cv::Mat image_pfm;

  if (file.is_open()) {
    // Read the type of file plus the 0x0a UNIX return character at the end
    char type[3];
    file.read(type, 3 * sizeof(char));

    // Read the width and height
    unsigned int width(0), height(0);
    file >> width >> height;

    // Read the 0x0a UNIX return character at the end
    char endOfLine;
    file.read(&endOfLine, sizeof(char));

    int num_com = 0;
    // The type gets the number of color channels
    if (type[1] == 'F') {
      image_pfm = cv::Mat(height, width, CV_32FC3);
      num_com = 3;
    } else if (type[1] == 'f') {
      image_pfm = cv::Mat(height, width, CV_32FC1);
      num_com = 1;
    }

    // Read the endianness plus the 0x0a UNIX return character at the end
    // Byte Order contains -1.0 or 1.0
    char byteOrder[4];
    file.read(byteOrder, 4 * sizeof(char));

    // Find the last line return 0x0a before the pixels of the image
    char findReturn = ' ';
    while (findReturn != 0x0a) {
      file.read(&findReturn, sizeof(char));
    }

    // Read each RGB colors as 3 floats and store it in the image.
    float* color = new float[num_com];
    for (unsigned int i = 0; i < height; ++i) {
      for (unsigned int j = 0; j < width; ++j) {
        file.read((char*)color, num_com * sizeof(float));

        // In the PFM fotmat the image is upside down
        if (num_com == 3) {
          image_pfm.at<cv::Vec3f>(height - 1 - i, j) =
              cv::Vec3f(color[2], color[1], color[0]);
        } else if (num_com == 1) {
          image_pfm.at<float>(height - 1 - i, j) = color[0];
        }
      }
    }

    delete[] color;

    // Close file
    file.close();
  } else {
    std::cout << "Could not open the file: " << file_path << std::endl;
  }

  return image_pfm;
}

bool SavePFM(const cv::Mat image, const std::string& file_path) {
  // Open the file as binary!
  std::ofstream imageFile(file_path.c_str(),
                          std::ios::out | std::ios::trunc | std::ios::binary);

  if (imageFile) {
    int width(image.cols), height(image.rows);
    int numberOfComponents(image.channels());

    // Write the type of the PFM file and ends by a line return
    char type[3];
    type[0] = 'P';
    type[2] = 0x0a;

    if (numberOfComponents == 3) {
      type[1] = 'F';
    } else if (numberOfComponents == 1) {
      type[1] = 'f';
    }

    imageFile << type[0] << type[1] << type[2];

    // Write the width and height and ends by a line return
    imageFile << width << " " << height << type[2];

    // Assumes little endian storage and ends with a line return 0x0a
    // Stores the type
    char byteOrder[10];
    byteOrder[0] = '-';
    byteOrder[1] = '1';
    byteOrder[2] = '.';
    byteOrder[3] = '0';
    byteOrder[4] = '0';
    byteOrder[5] = '0';
    byteOrder[6] = '0';
    byteOrder[7] = '0';
    byteOrder[8] = '0';
    byteOrder[9] = 0x0a;

    for (int i = 0; i < 10; ++i) {
      imageFile << byteOrder[i];
    }

    // Store the floating points RGB color upside down, left to right
    float* buffer = new float[numberOfComponents];

    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        if (numberOfComponents == 1) {
          buffer[0] = image.at<float>(height - 1 - i, j);
        } else {
          cv::Vec3f color = image.at<cv::Vec3f>(height - 1 - i, j);

          // OpenCV stores as BGR
          buffer[0] = color.val[2];
          buffer[1] = color.val[1];
          buffer[2] = color.val[0];
        }

        // Write the values
        imageFile.write((char*)buffer, numberOfComponents * sizeof(float));
      }
    }

    delete[] buffer;

    imageFile.close();
  } else {
    std::cerr << "Could not open the file : " << file_path << std::endl;
    return false;
  }

  return true;
}

}  // namespace colmap
