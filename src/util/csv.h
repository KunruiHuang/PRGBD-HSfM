#ifndef COLMAP_CSV_H
#define COLMAP_CSV_H

#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace colmap {
namespace csv {

class Error : public std::runtime_error {
 public:
  Error(const std::string& msg)
      : std::runtime_error(std::string("CSVparser : ").append(msg)) {}
};

class Row {
 public:
  Row(const std::vector<std::string>& header, char sep = ',');
  ~Row(void);

 public:
  unsigned int size(void) const;
  void push(const std::string&);
  bool set(const std::string&, const std::string&);
  std::vector<std::string> values();

 private:
  const char _sep;
  std::vector<std::string> _header;
  std::vector<std::string> _values;

 public:
  template <typename T>
  const T getValue(unsigned int pos) const {
    if (pos < _values.size()) {
      T res;
      std::stringstream ss;
      ss << _values[pos];
      ss >> res;
      return res;
    }
    throw Error("can't return this value (doesn't exist)");
  }
  inline std::vector<std::string>& getHeader();
  const std::string operator[](unsigned int) const;
  const std::string operator[](const std::string& valueName) const;
  friend std::ostream& operator<<(std::ostream& os, const Row& row);
  friend std::ofstream& operator<<(std::ofstream& os, const Row& row);
};

enum DataType { eFILE = 0, ePURE = 1 };

class Process {
 public:
  Process(const std::string&, const DataType& type = eFILE, char sep = ',');
  Process(const std::string&, const std::vector<std::string>& header,
          const DataType& type = eFILE, char sep = ',');
  ~Process(void);

 public:
  Row& getRow(unsigned int row) const;
  unsigned int rowCount(void) const;
  unsigned int columnCount(void) const;
  std::vector<std::string>& getHeader(void);
  const std::vector<std::string>& getHeader(void) const;
  bool modifiedHeader(const std::vector<std::string>& new_header);
  bool modifiedHeader(unsigned int pos, const std::string& head);
  const std::string getHeaderElement(unsigned int pos) const;
  const std::string& getFileName(void) const;

 public:
  bool deleteRow(unsigned int row);
  bool addRow(unsigned int pos, const std::vector<std::string>&);
  bool pushRow(const std::vector<std::string>&);
  void sync(void) const;

 protected:
  void processHeader(void);
  void processContent(void);

 private:
  std::string _file;
  const DataType _type;
  const char _sep;
  std::vector<std::string> _originalFile;
  std::vector<std::string> _header;
  std::vector<Row*> _content;

 public:
  Row& operator[](unsigned int row) const;
};

}  // namespace csv
} // namespace colmap

#endif  // COLMAP_CSV_H
