#include "util/csv.h"

namespace colmap {
namespace csv {

Process::Process(const std::string& data, const DataType& type, char sep)
    : _type(type), _sep(sep) {
  std::string line;
  if (type == eFILE) {
    _file = data;
    std::ifstream ifile(_file.c_str());
    if (ifile.is_open()) {
      while (ifile.good()) {
        getline(ifile, line);
        // Remove possible _sep at the end
        line.erase(line.find_last_not_of(_sep) + 1);
        if (!line.empty()) _originalFile.push_back(line);
      }
      ifile.close();

      if (_originalFile.empty())
        throw Error(std::string("No Data in ").append(_file));

      processHeader();
      processContent();
    } else
      throw Error(std::string("Failed to open ").append(_file));
  } else {
    std::istringstream stream(data);
    while (std::getline(stream, line))
      if (!line.empty()) _originalFile.push_back(line);
    if (_originalFile.empty())
      throw Error(std::string("No Data in pure content"));

    processHeader();
    processContent();
  }
}

Process::Process(const std::string& filename,
                 const std::vector<std::string>& header, const DataType& type,
                 char sep)
    : _file(filename), _type(type), _sep(sep), _header(header) {}

Process::~Process(void) {
  std::vector<Row*>::iterator it;

  for (it = _content.begin(); it != _content.end(); it++) delete *it;
}

void Process::processHeader(void) {
  std::stringstream ss(_originalFile[0]);
  std::string item;

  while (std::getline(ss, item, _sep)) {
    _header.emplace_back(item);
  }
}

void Process::processContent(void) {
  std::vector<std::string>::iterator it;

  it = _originalFile.begin();
  it++;  // skip header

  for (; it != _originalFile.end(); it++) {
    bool quoted = false;
    auto tokenStart = 0;
    auto i = 0;

    Row* row = new Row(_header);

    for (; i != (int)it->length(); i++) {
      if (it->at(i) == '"') {
        quoted = ((quoted) ? (false) : (true));
      } else if (it->at(i) == _sep && !quoted) {
        row->push(it->substr(tokenStart, i - tokenStart));
        tokenStart = i + 1;
      }
    }

    // end
    row->push(it->substr(tokenStart, it->length() - tokenStart));
    while (row->size() < _header.size()) row->push("");

    // if value(s) missing
    if (row->size() != _header.size()) throw Error("corrupted data !");
    _content.push_back(row);
  }
}

Row& Process::getRow(unsigned int rowPosition) const {
  if (rowPosition < _content.size()) return *(_content[rowPosition]);
  throw Error("can't return this row (doesn't exist)");
}

Row& Process::operator[](unsigned int rowPosition) const {
  return Process::getRow(rowPosition);
}

unsigned int Process::rowCount(void) const { return _content.size(); }

unsigned int Process::columnCount(void) const { return _header.size(); }

std::vector<std::string>& Process::getHeader(void) { return _header; }

const std::vector<std::string>& Process::getHeader(void) const {
  return _header;
}

bool Process::modifiedHeader(const std::vector<std::string>& new_header) {
  if (new_header.size() != _header.size()) return false;
  _header = new_header;
  // 同时直接修改一下所有行的文件头
  for (auto& content : _content) {
    content->getHeader() = new_header;
  }
  return true;
}

bool Process::modifiedHeader(unsigned int pos, const std::string& head) {
  if (pos < 0 || pos > _header.size()) return false;
  _header[pos] = head;
  return true;
}

const std::string Process::getHeaderElement(unsigned int pos) const {
  if (pos >= _header.size())
    throw Error("can't return this header (doesn't exist)");
  return _header[pos];
}

bool Process::deleteRow(unsigned int pos) {
  if (pos < _content.size()) {
    delete *(_content.begin() + pos);
    _content.erase(_content.begin() + pos);
    return true;
  }
  return false;
}

bool Process::addRow(unsigned int pos, const std::vector<std::string>& r) {
  Row* row = new Row(_header);

  for (auto it = r.begin(); it != r.end(); it++) row->push(*it);

  if (pos <= _content.size()) {
    _content.insert(_content.begin() + pos, row);
    return true;
  }
  return false;
}

bool Process::pushRow(const std::vector<std::string>& r) {
  Row* row = new Row(_header, _sep);
  if (_header.size() != r.size()) {
    std::cerr << "The length of the content is inconsistent with the header."
              << std::endl;
    return false;
  }
  for (auto it = r.begin(); it != r.end(); it++) row->push(*it);
  _content.emplace_back(row);

  return true;
}

void Process::sync(void) const {
  if (_type == DataType::eFILE) {
    std::ofstream f;
    f.open(_file, std::ios::out | std::ios::trunc);

    // header
    unsigned int i = 0;
    for (auto it = _header.begin(); it != _header.end(); it++) {
      f << *it;
      if (i < _header.size() - 1)
        f << _sep;
      else
        f << std::endl;
      i++;
    }

    for (auto it = _content.begin(); it != _content.end(); it++)
      f << **it << std::endl;
    f.close();
    std::cout << "csv file synchronization completed successfully."
              << std::endl;
  }
}

const std::string& Process::getFileName(void) const { return _file; }

// Row
Row::Row(const std::vector<std::string>& header, char sep)
    : _sep(sep), _header(header) {}

Row::~Row(void) {}

unsigned int Row::size(void) const { return _values.size(); }

void Row::push(const std::string& value) { _values.push_back(value); }

bool Row::set(const std::string& key, const std::string& value) {
  std::vector<std::string>::const_iterator it;
  int pos = 0;

  for (it = _header.begin(); it != _header.end(); it++) {
    if (key == *it) {
      _values[pos] = value;
      return true;
    }
    pos++;
  }
  return false;
}

std::vector<std::string> Row::values() { return this->_values; }

std::vector<std::string>& Row::getHeader() { return _header; }

const std::string Row::operator[](unsigned int valuePosition) const {
  if (valuePosition < _values.size()) return _values[valuePosition];
  throw Error("can't return this value (doesn't exist)");
}

const std::string Row::operator[](const std::string& key) const {
  std::vector<std::string>::const_iterator it;
  int pos = 0;

  for (it = _header.begin(); it != _header.end(); it++) {
    if (key == *it) return _values[pos];
    pos++;
  }

  throw Error("can't return this value (doesn't exist)");
}

std::ostream& operator<<(std::ostream& os, const Row& row) {
  for (unsigned int i = 0; i != row._values.size(); i++)
    os << row._values[i] << " | ";

  return os;
}

std::ofstream& operator<<(std::ofstream& os, const Row& row) {
  for (unsigned int i = 0; i != row._values.size(); i++) {
    os << row._values[i];
    if (i < row._values.size() - 1) os << row._sep;
  }
  return os;
}

}  // namespace csv
} // namespace colmap