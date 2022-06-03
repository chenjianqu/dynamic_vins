/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_IO_UTILS_H
#define DYNAMIC_VINS_IO_UTILS_H

#include <filesystem>

#include "utils/def.h"



namespace dynamic_vins{\

void ClearDirectory(const string &path);

vector<fs::path> GetDirectoryFileNames(const string &path);


}


#endif //DYNAMIC_VINS_IO_UTILS_H
