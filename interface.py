# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 11:50:37 2023

@author: rmgu

Copyright (C) 2023  DHI

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from marshmallow import Schema
from webargs import fields


class Inputs(Schema):
    aoi_name = fields.String(required=True)
    date = fields.Date(required=True)
    spatial_res = fields.String(required=False, missing="s2")
    temporal_res = fields.String(required=False, missing="dekadal")
    ftp_url = fields.String(required=True)
    ftp_port = fields.Integer(required=True)
    ftp_username = fields.String(required=True)
    ftp_pass = fields.String(required=True)


class Outputs(Schema):
    output_path = fields.List(fields.String, required=True)
