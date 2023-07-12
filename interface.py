# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 11:50:37 2023

@author: rmgu
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
