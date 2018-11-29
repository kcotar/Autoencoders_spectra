import imp, os

from astropy.table import Table, join, unique, vstack
from glob import glob

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path

outliers = []


class PointSelector:
    def __init__(self, ax, x_val, y_val):
        self.x = x_val
        self.y = y_val
        self.n_selection = 1
        self.lasso = LassoSelector(ax, self.determine_points)

    def determine_points(self, vert):
        self.vertices = vert
        if len(self.vertices) > 0:
            self.path = Path(self.vertices)
            # determine objects in region
            print 'Determining objects in selected region'
            temp = [a_row['sobject_id'] for a_row in abund_param if self.path.contains_point((a_row[self.x], a_row[self.y]))]
            self.n_selected = len(temp)
            if self.n_selection > 0:
                idx_ok = np.in1d(abund_param['sobject_id'], temp)
                idx_bad = np.logical_and(~idx_ok, np.isfinite(abund_param[self.y]))
                outliers.append([s for s in abund_param['sobject_id'][idx_bad]])
                print abund_param[idx_bad]['sobject_id']
            else:
                print 'Number of points in region is too small'
        else:
            print 'Number of vertices in selection is too small'


# --------------------------------------------------------
# ---------------- Data reading --------------------------
# --------------------------------------------------------
galah_data_input = '/data4/cotar/'
abund_param_file = 'GALAH_iDR3_ts_DR2.fits'  # can have multiple lines with the same sobject_id - this is on purpose

abund_param = Table.read(galah_data_input + abund_param_file)
abund_param = unique(abund_param, keys=['sobject_id'])
sme_abundances_list = [col for col in abund_param.colnames if '_fe' in col and len(col.split('_')) == 2 and len(col.split('_')[0]) <= 2]
sme_params = ['teff', 'fe_h', 'logg', 'vbroad']

# validation cluster data
openc_param = Table.read(galah_data_input + 'GALAH_iDR3_OpenClusters.fits')
globc_param = Table.read(galah_data_input + 'GALAH_iDR3_GlobularClusters.fits')
cluster_param = vstack([openc_param, globc_param])

# TEST stact all data together as train
remove = [131118001101089,131118001101102,131118001101397,131118001901187,131118001901243,131118001901295,131118001901332,131120001101152,131120001101210,131120001101295,131120001401356,131120001401384,131120001401388,131120002001005,131120002001038,131120002001086,131120002001120,131120002001135,131120002001145,131120002001173,131120002001180,131120002001185,131120002001191,131120002001196,131120002001209,131120002001211,131120002001252,131120002001276,131120002001282,131120002001295,131120002001305,131120002001310,131120002001312,131120002001333,131120002001376,131120002001377,131123001701022,131123001701049,131123001701109,131123001701159,131123001701178,131123001701217,131123001701261,131123001701269,131123001701270,131123001701282,131123001701286,131123001701290,131123001701291,131123001701292,131123001701293,131123001701305,131123001701314,131123001701330,131123001701333,131123001701337,131123001701339,131123001701342,131123001701343,131123001701344,131123001701345,131123001701353,131123001701356,131123001701362,131123001701364,131123001701368,131123001701369,131123001701373,131123001701378,131123001701379,131123002001214,131123002001243,131123002001279,131123002001363,131123002001373,131123002501247,131216001601012,131216001601366,131216001601373,131217003301122,131217003901226,131217003901244,131217003901339,131217003901349,131217004401069,131217004401212,131217004401241,131217004401341,131217004401366,131217004401382,131217004401384,131217004401393,131218003001006,131218003001021,131218003001041,131218003001085,131218003001090,131218003001124,131218003001163,131218003001260,131218003001272,131218003001277,131218003001304,131218003001313,131218003001323,131218003001332,131218003001349,131218003001367,131218003001372,131218003001373,131218003001375,131218003001379,131218003001380,131218003001387,131218003001394,131220001801008,131220001801035,131220001801041,131220001801047,131220001801054,131220001801058,131220001801097,131220001801146,131220001801149,131220001801157,131220001801160,131220001801165,131220001801178,131220001801215,131220001801217,131220001801286,131220001801291,131220001801314,131220001801320,131220001801330,131220001801343,131220001801345,131220001801353,131220001801356,131220001801362,131220001801363,131220001801368,131220001801370,131220001801373,131220001801382,131220001801383,131220001801394,131220001801395,140111001601017,140111001601047,140111001601054,140111001601065,140111001601101,140111001601148,140111001601149,140111001601166,140111001601176,140111001601180,140111001601195,140111001601261,140111001601270,140111001601280,140111001601283,140111001601288,140111001601290,140111001601292,140111001601293,140111001601298,140111001601304,140111001601305,140111001601306,140111001601310,140111001601313,140111001601314,140111001601318,140111001601323,140111001601332,140111001601337,140111001601345,140111001601349,140111001601356,140111001601357,140111001601358,140111001601363,140111001601364,140111001601366,140111001601370,140111001601371,140111001601375,140111001601378,140111001601380,140111001601387,140111001601388,140111001601389,140111001601392,140111002101293,140111002101301,140111002101307,140111002101317,140111002101327,140111002101336,140111002101375,140111002601041,140111002601042,140111002601061,140111002601073,140111002601097,140111002601101,140111002601125,140111002601163,140111002601232,140111002601255,140111002601258,140111002601295,140111002601296,140111002601323,140111002601340,140111002601342,140111002601343,140111002601344,140111002601346,140111002601352,140111002601374,140111002601375,140111002601383,140111002601388,140112001801007,140112001801011,140112001801020,140112001801022,140112001801024,140112001801026,140112001801028,140112001801033,140112001801035,140112001801038,140112001801043,140112001801047,140112001801048,140112001801049,140112001801052,140112001801053,140112001801065,140112001801066,140112001801107,140112001801110,140112001801117,140112001801121,140112001801128,140112001801148,140112001801149,140112001801152,140112001801155,140112001801158,140112001801165,140112001801175,140112001801179,140112001801192,140112001801209,140112001801210,140112001801219,140112001801220,140112001801272,140112001801274,140112001801275,140112001801286,140112001801287,140112001801288,140112001801290,140112001801294,140112001801299,140112001801303,140112001801304,140112001801306,140112001801307,140112001801308,140112001801309,140112001801315,140112001801316,140112001801317,140112001801322,140112001801323,140112001801324,140112001801325,140112001801332,140112001801337,140112001801343,140112001801345,140112001801349,140112001801352,140112001801362,140112001801369,140112001801371,140112001801372,140112001801375,140112001801378,140112001801379,140112001801380,140112001801387,140112001801388,140112001801389,140112001801390,140112001801392,140112001801394,140112002801003,140112002801013,140112002801031,140112002801034,140112002801041,140112002801061,140112002801097,140112002801099,140112002801101,140112002801104,140112002801125,140112002801133,140112002801147,140112002801163,140112002801168,140112002801170,140112002801214,140112002801221,140112002801224,140112002801230,140112002801232,140112002801236,140112002801239,140112002801241,140112002801258,140112002801283,140112002801295,140112002801296,140112002801335,140112002801352,140112002801374,140112002801375,140112002801388,140112002801394,140113002901004,140113002901011,140113002901020,140113002901021,140113002901022,140113002901023,140113002901026,140113002901028,140113002901033,140113002901034,140113002901035,140113002901037,140113002901038,140113002901040,140113002901043,140113002901048,140113002901049,140113002901052,140113002901066,140113002901107,140113002901114,140113002901121,140113002901128,140113002901144,140113002901148,140113002901152,140113002901155,140113002901158,140113002901165,140113002901175,140113002901179,140113002901192,140113002901206,140113002901209,140113002901210,140113002901212,140113002901215,140113002901219,140113002901220,140113002901238,140113002901275,140113002901294,140113002901297,140113002901299,140113002901303,140113002901304,140113002901307,140113002901308,140113002901309,140113002901315,140113002901316,140113002901317,140113002901324,140113002901325,140113002901332,140113002901343,140113002901352,140113002901364,140113002901366,140113002901371,140113002901372,140113002901375,140113002901378,140113002901379,140113002901387,140113002901388,140113002901389,140113002901390,140113002901392,140113002901394,140114002401046,140114002401073,140114002401095,140114002401098,140114002401114,140114002401146,140114002401151,140114002401164,140114002401186,140114002401203,140114002401204,140114002401209,140114002401236,140114002401239,140114002401249,140114002401257,140114002401275,140114002401290,140114002401299,140114002401318,140114002401331,140114002401335,140114002401340,140114002401345,140114002401355,140114002401376,140114002401378,140114002401390,140114002401397,140114002401398,140114004201126,140115002101045,140115002101046,140115002101114,140115002101146,140115002101164,140115002101186,140115002101203,140115002101236,140115002101239,140115002101249,140115002101257,140115002101318,140115002101331,140115002101340,140115002101345,140115002101355,140115002101376,140115002101378,140115002101390,140115002101397,140115003901103,140115003901112,140115003901210,140115003901230,140115003901232,140115003901242,140115003901269,140115003901285,140115003901314,140115003901329,140115003901346,140115003901380,140115003901386,140116002201046,140116002201080,140116002201095,140116002201146,140116002201186,140116002201204,140116002201227,140116002201236,140116002201239,140116002201249,140116002201290,140116002201299,140116002201324,140116002201331,140116002201334,140116002201335,140116002201340,140116002201345,140116002201376,140116002201378,140116002201390,140116002201397,140116002201398,140116003701142,140117001501355,140117003101030,140117003101210,140117003101242,140117003101285,140117003101314,140117003101329,140117003101346,140117003101380,140117003101397,140118002001167,140118002501319,140118002501332,140118002501338,140118002501348,140118003001332,140118003001334,140209001701110,140209003201070,140209003201202,140209003201213,140209003201242,140209003201258,140209003201353,140209003201385,140209004201321,140303000401111,140303002001019,140303002001068,140303002001076,140303002001102,140303002001106,140303002001112,140303002001122,140303002001147,140303002001148,140303002001155,140303002001169,140303002001202,140303002001230,140303002001231,140303002001232,140303002001236,140303002001275,140303002001293,140304001801013,140304001801016,140304001801017,140304001801019,140304001801021,140304001801029,140304001801037,140304001801038,140304001801041,140304001801049,140304001801066,140304001801068,140304001801076,140304001801079,140304001801085,140304001801094,140304001801095,140304001801097,140304001801102,140304001801106,140304001801112,140304001801122,140304001801144,140304001801147,140304001801148,140304001801154,140304001801155,140304001801167,140304001801169,140304001801192,140304001801202,140304001801206,140304001801208,140304001801213,140304001801230,140304001801231,140304001801236,140304001801238,140304001801275,140304001801279,140304001801293,140304001801335,140304001801337,140304001801341,140304001801356,140304001801362,140304001801367,140304001801394,140305003201003,140305003201009,140305003201014,140305003201016,140305003201025,140305003201028,140305003201036,140305003201038,140305003201046,140305003201051,140305003201057,140305003201061,140305003201082,140305003201099,140305003201101,140305003201107,140305003201108,140305003201109,140305003201110,140305003201112,140305003201123,140305003201130,140305003201131,140305003201133,140305003201135,140305003201139,140305003201144,140305003201145,140305003201147,140305003201152,140305003201154,140305003201155,140305003201156,140305003201157,140305003201158,140305003201164,140305003201166,140305003201169,140305003201170,140305003201178,140305003201184,140305003201185,140305003201187,140305003201194,140305003201196,140305003201198,140305003201203,140305003201234,140305003201258,140305003201261,140305003201265,140305003201266,140305003201267,140305003201268,140305003201270,140305003201274,140305003201276,140305003201287,140305003201291,140305003201305,140305003201306,140305003201322,140305003201323,140305003201326,140305003201327,140305003201328,140305003201329,140305003201330,140305003201336,140305003201337,140305003201342,140305003201349,140305003201353,140305003201361,140305003201389,140305003201393,140305003201399,140305003701004,140305003701013,140305003701017,140305003701019,140305003701025,140305003701037,140305003701040,140305003701041,140305003701049,140305003701063,140305003701066,140305003701068,140305003701079,140305003701085,140305003701097,140305003701102,140305003701106,140305003701112,140305003701122,140305003701139,140305003701143,140305003701147,140305003701148,140305003701155,140305003701192,140305003701208,140305003701213,140305003701220,140305003701236,140305003701238,140305003701279,140305003701293,140305003701316,140305003701337,140305003701356,140305003701362,140305003701377,140307001101114,140307001101204,140307001101205,140307001601242,140307002601004,140307002601006,140307002601013,140307002601014,140307002601016,140307002601017,140307002601019,140307002601021,140307002601025,140307002601029,140307002601035,140307002601037,140307002601038,140307002601040,140307002601041,140307002601042,140307002601045,140307002601063,140307002601066,140307002601076,140307002601079,140307002601084,140307002601085,140307002601095,140307002601097,140307002601102,140307002601105,140307002601106,140307002601112,140307002601122,140307002601125,140307002601139,140307002601143,140307002601144,140307002601147,140307002601148,140307002601154,140307002601155,140307002601157,140307002601159,140307002601161,140307002601164,140307002601169,140307002601180,140307002601192,140307002601206,140307002601208,140307002601213,140307002601220,140307002601225,140307002601227,140307002601232,140307002601234,140307002601236,140307002601238,140307002601247,140307002601256,140307002601257,140307002601275,140307002601279,140307002601286,140307002601290,140307002601293,140307002601316,140307002601322,140307002601334,140307002601335,140307002601337,140307002601338,140307002601341,140307002601356,140307002601359,140307002601367,140307002601375,140307002601377,140307002601382,140307002601392,140307002601394,140307003101003,140307003101012,140307003101023,140307003101052,140307003101055,140307003101057,140307003101059,140307003101087,140307003101095,140307003101110,140307003101138,140307003101149,140307003101172,140307003101208,140307003101218,140307003101249,140307003101269,140307003101282,140307003101287,140307003101289,140307003101303,140307003101307,140307003101311,140307003101324,140307003101339,140307003101361,140307003101362,140307003101375,140307003101379,140307003101395,140309003601073,140310002101361,140311007101283,140311009101170,140312003501057,140312003501144,140312004501327,140313004701361,140314002601264,140314002601362,140314004001171,140314005201004,140314005201011,140314005201015,140314005201019,140314005201026,140314005201038,140314005201057,140314005201058,140314005201067,140314005201070,140314005201080,140314005201086,140314005201093,140314005201094,140314005201098,140314005201115,140314005201120,140314005201139,140314005201144,140314005201151,140314005201152,140314005201153,140314005201157,140314005201158,140314005201163,140314005201173,140314005201174,140314005201176,140314005201177,140314005201184,140314005201185,140314005201187,140314005201190,140314005201195,140314005201207,140314005201209,140314005201210,140314005201214,140314005201217,140314005201224,140314005201231,140314005201234,140314005201264,140314005201269,140314005201291,140314005201296,140314005201301,140314005201307,140314005201312,140314005201326,140314005201363,140314005201365,140314005201379,140314005201392,140314005201394,140315002501003,140315002501004,140315002501005,140315002501015,140315002501019,140315002501026,140315002501035,140315002501048,140315002501054,140315002501057,140315002501058,140315002501068,140315002501070,140315002501080,140315002501086,140315002501094,140315002501098,140315002501115,140315002501120,140315002501139,140315002501144,140315002501151,140315002501152,140315002501153,140315002501157,140315002501158,140315002501163,140315002501176,140315002501177,140315002501184,140315002501185,140315002501187,140315002501190,140315002501193,140315002501209,140315002501210,140315002501214,140315002501224,140315002501226,140315002501231,140315002501234,140315002501237,140315002501259,140315002501264,140315002501269,140315002501271,140315002501291,140315002501294,140315002501312,140315002501321,140315002501326,140315002501336,140315002501349,140315002501363,140315002501365,140315002501379,140315002501380,140315002501392,140315002501394,140315002501396,140316004201004,140316004201005,140316004201011,140316004201019,140316004201026,140316004201035,140316004201042,140316004201048,140316004201054,140316004201058,140316004201067,140316004201068,140316004201080,140316004201093,140316004201094,140316004201098,140316004201112,140316004201120,140316004201133,140316004201139,140316004201144,140316004201151,140316004201157,140316004201158,140316004201163,140316004201173,140316004201174,140316004201176,140316004201177,140316004201179,140316004201195,140316004201198,140316004201204,140316004201207,140316004201209,140316004201210,140316004201224,140316004201226,140316004201231,140316004201234,140316004201237,140316004201240,140316004201264,140316004201269,140316004201271,140316004201283,140316004201291,140316004201296,140316004201301,140316004201307,140316004201312,140316004201321,140316004201326,140316004201331,140316004201363,140316004201371,140316004201379,140316004201380,140316004201384,140316004201392,140316004201394,140409003601270,140412000201084,140412000201189,140412000201210,140412000201284,140412001201054,140412002201076,140413002201112,140413002201199,140413002201387,140413003701131,140413003701272,140414002001174,140414002001214,140414002001284,140414002601178,140414003101209,140414004101195,140607000701031,140607000701227,140607000701281,140607000701366,140607002001133,140608001401028,140608001901248,140608004301069,140609001101033,140609001601118,140609002101242,140610003901043,140610004401266,140610005001043,140610005001057,140611001601366,140611003001163,140707002101003,140707002601055,140707003101110,140707003101148,140707003601047,140708001201066,140708001701306,140708002701325,140709003001083,140709003001251,140710000101192,140710000801284,140710001701284,140710002501284,140711001301216,140711001301321,140711001301368,140711001301387,140711001901263,140711003401086,140713002401298,140713002401341,140713002401344,140805002101221,140805002601213,140805003601171,140805003601251,140805004201105,140805004201241,140805004201252,140805004801143,140806002301328,140806002901026,140806002901066,140806003501103,140806004101126,140806004701399,140807005001034,140807005601198,140808002101079,140808003201002,140808003701149,140808004201016,140808004201023,140808004201047,140808004201086,140808004201090,140808004201108,140808004201114,140808004201138,140808004201139,140808004201167,140808004201168,140808004201169,140808004201194,140808004201205,140808004201217,140808004201232,140808004201251,140808004201268,140808004201355,140808004701162,140809001601079,140809001601140,140809001601356,140809003101229,140810002201109,140810005301313,140811002101132,140812003201070,140812003801313,140813001601170,140813002201136,140813002201153,140813003001122,140814003301057,140814004301144,140814006001053,140823002701004,140823002701005,140823002701012,140823002701388,140823002701393,140823002701397,140823002701398,140824004801004,140824004801005,140824004801026,140824004801039,140824004801086,140824004801115,140824004801125,140824004801193,140824004801199,140824004801206,140824004801209,140824004801215,140824004801266,140824004801284,140824004801311,140824004801322,140824004801328,140824004801334,140824004801335,140824004801336,140824004801339,140824004801340,140824004801341,140824004801347,140824004801348,140824004801349,140824004801354,140824004801355,140824004801363,140824004801371,140824004801372,140824004801379,140824004801386,140824004801392,140824004801393,140824004801394,140824004801396,140824004801397,140824004801398,140824004801399,140824005301140,140824006301074,141031003601116,141102002401174,141102002401298,141102003201281,141103002601208,141104002301153,141104002301245,141104002301258,141231004601038,141231004601139,150101002501146,150101002901099,150102002701185,150102002701210,150102003201103,150103003001191,150103003001313,150103003501039,150103004001242,150106003201256,150106003201298,150107004701189,150109001001008,150109001001031,150109001001049,150109001001080,150109001001142,150109001001311,150109001001355,150204001601066,150204002101256,150204002401079,150204002901187,150204003701096,150205005001119,150207002101172,150207002601080,150207003601110,150207004101145,150207004101258,150207004601316,150207005101083,150207005101129,150207005101132,150207005101280,150207005101302,150208004701041,150210003201013,150210003201064,150210003201161,150210003701391,150210004201233,150210004201260,150210004201351,150330001701014,150330001701099,150401003601047,150401004101204,150408002901015,150408004101125,150408004701333,150409001601067,150409002101075,150409002101101,150409002601298,150409003601284,150409003601360,150409004101156,150409005601160,150409005601257,150410003801124,150410003801178,150410003801208,150411003101112,150411003601103,150411004101326,150411004601208,150411005101279,150411005101327,150411005601139,150411006101249,150411006601028,150412002101126,150412002601330,150412003601233,150412004601264,150412005601291,150412006101126,150413003601347,150413004101121,150428000101180,150428000601158,150428001101339,150429001601331,150429002101291,150430002301295,150430003301326,150601001601081,150601001601192,150601002401302,150601004801134,150601004801244,150602001601171,150602002101376,150602003301024,150602003301295,150602003301347,150604003401385,150607005601381,150607006101073,150703002601134,150703003101056,150703003101244,150703003101260,150703004101038,150705001901153,150705003901229,150705003901233,150705004401228,150718004401364,150824002101176,150824002101264,150824002601339,150824002601341,150824002601343,150827003401378,150827004001280,150828005701074,150829002601255,150829002601264,150829005701066,150829005701125,150829005701181,150830002301006,150830002301386,150830002801278,150830004001108,150830005101013,150830006601022,150830006601106,150830006601111,150830006601134,150830006601135,150830006601171,150830006601218,150831004001241,150831004001307,150831005001140,150831005001321,150901000601043,150901000601170,150901001101002,150901001101022,150901001101030,150901001101042,150901001101128,150901001101141,150901001101142,150901001101144,150901001101158,150901001101166,150901001101184,150901001101186,150901001101245,150901001101274,150901001101278,150901001101281,150901001101286,150901001101294,150901001101305,150901001101324,150901001101340,150901001101394,150901001101399,150901002401031,151008002101024,151008004001047,151008004001159,151008004001204,151008004001216,151008004001247,151009001601257,151009001601262,151009004601256,151009004601308,151009005101033,151009005101153,151109002601393,151110003101279,151111001601286,151111003101251,151111003601136,151111003601194,151111004301279,151111004301287,151111004301294,151111004301326,151111004301327,151219004601012,151224002101119,151224002101127,151224002101275,151224002101330,151227004201003,151227004201022,151227004201135,151227004201158,151227004201295,151227004201352,151227004701006,151227005701195,151227005701238,151230002201345,151231002601118,151231003701005,151231003701006,151231003701008,151231003701032,151231003701052,151231003701054,151231003701065,151231003701090,151231003701306,151231003701326,151231003701336,151231003701378,151231004301114,151231004301135,151231004901051,160106001601049,160106001601063,160106001601095,160106001601242,160106001601249,160106001601261,160106001601268,160106001601306,160106001601343,160106001601371,160106004101038,160106004101267,160106004101272,160106004101274,160106004101276,160106004101277,160106004101279,160106004101283,160106004101287,160106004101292,160106004101293,160106004101297,160106004101299,160106004101304,160106004101307,160106004101310,160106004101316,160106004101323,160106004101341,160106004101356,160107001601136,160107004101279,160107004101288,160107004101294,160107004101296,160107004101309,160107004101311,160107004101320,160107004101361,160108002001031,160108002001157,160108002001262,160108002001342,160108002001386,160108002001391,160108002001393,160108003601365,160109002001238,160109002501106,160109002501255,160110002101224,160110003101037,160110003101390,160111001601014,160111001601095,160111002101290,160111002101381,160112001601226,160113001601011,160113002101030,160113002101045,160113002101047,160113002101053,160113002101058,160113002101059,160113002101074,160113002101079,160113002101102,160113002101109,160113002101139,160113002101143,160113002101167,160113002101170,160113002101177,160113002101184,160113002101204,160113002101207,160113002101256,160113002101269,160113002101329,160113002101342,160113002101385,160113002101387,160113002901197,160113002901241,160113002901244,160113002901246,160123002601178,160123003101311,160124002601265,160125003501038,160125003501211,160125003501335,160125004501223,160129003601106,160129003601159,160129004201011,160130003601246,160130004101294,160130004601148,160130006301225,160325003701262,160325004201151,160326000601204,160327004101194,160327005101029,160327005101138,160327006101102,160327006601044,160327006601052,160327006601053,160327006601054,160327006601055,160327006601056,160327006601060,160327006601062,160327006601243,160327006601382,160328000101191,160330001601262,160331001701173,160331002201099,160331003201214,160331004801099,160331005801143,160401002101020,160401002101055,160401002101126,160401002101163,160401002101234,160401002101328,160401002101341,160401004401280,160402003601165,160402003601261,160402005101028,160402005601024,160402005601036,160402005601213,160402005601260,160402005601352,160402006601188,160402006601246,160403002501105,160403002501241,160403004201209,160403004201235,160403004201242,160403004201295,160403004701315,160403004701363,160415001601234,160415001601316,160417002201129,160418003101058,160418003601175,160418004601183,160419002601267,160419003601125,160419003601126,160419003601131,160419003601136,160419003601137,160419003601140,160419003601144,160419003601164,160419004101365,160419004601021,160419005101043,160419005101046,160419005101051,160419005101059,160419005101242,160419005701099,160419005701369,160420003801395,160420003801397,160421001601152,160421002101206,160421002101378,160421002601023,160421002601162,160421002601380,160421003601118,160421004601148,160421005601149,160422002001337,160422002501263,160422003501005,160422003501027,160423002201055,160423002201091,160423002201161,160423002201205,160423002201329,160423002201376,160423004401355,160424004701128,160424005701265,160425003401023,160426003501254,160426004001318,160426005501327,160426006701380,160426007401201,160513001601198,160513002101112,160513002601109,160514001801192,160522002101125,160522002101185,160522002101278,160522002601262,160522005101377,160522005601142,160522005601164,160522006101096,160522006101141,160522006101178,160522006601236,160524002101025,160524002101243,160524002701158,160524002701216,160524002701257,160524004901067,160525002201106,160525002201160,160525002201295,160527001601333,160529003401081,160529003401093,160529003401164,160529004201151,160529004801133,160530001601022,160530003301170,160530003301174,160530003301241,160530003901237,160530005001182,160530006001383,160531003101131,160531004101057,160531004101237,160531004101246,160531004101251,160531004101313,160531004101382,160531004601029,160531004601061,160531004601173,160531004601201,160602001601013,160602001601112,160602001601366,160724003501009,160811002901146,160811004601056,160811004601241,160812002101258,160813001601298,160815002101122,160815004301189,160816003201047,160816003701284,160817003601053,160916001801391,160916002301012,160916002801122,160916003301182,160916004301002,160919001601195,160919004001104,160923001801087,160923002501239,160923004201059,160923004201066,160923004201082,161006003101008,161006005401136,161006005401279,161007003301208,161008002001351,161009002601056,161009003201197,161009003201298,161009003801177,161009004801194,161011003401210,161011004001219,161011004001288,161013001601327,161013002601170,161013003801353,161013005401393,161105003601387,161106003101180,161106005101008,161106005101191,161107002801132,161107004401081,161118002601035,161118002601051,161118003501016,161118004001207,161119002801037,161119004201164,161210003401013,161210003401149,161210004201154,161211003101372,161212001601334,161212002601107,161212002601257,161212002601326,161212004101026,161212004101190,161212004101206,161212004101252,161212004101327,161212004101330,161212004101344,161212004101352,161212004601047,161212004601068,161212004601146,161212004601167,161212004601170,161212004601181,161212004601194,161212004601201,161212004601218,161212004601232,161212004601240,161212004601241,161212004601253,161212004601258,161212004601260,161212004601266,161212004601268,161212004601278,161212004601284,161212004601291,161212004601303,161212004601306,161212004601316,161212004601359,161212004601365,161212004601375,161212004601393,161213001601284,161213001601307,161213003601240,161213005101011,161217003101142,161217004101176,161217004101303,161217004601039,161217006101165,161218002101288,161218003101116,161218003101172,161218003101188,161219002601228,161219004101009,170105002101173,170107003101004,170107003101335,170107004201087,170108002701066,170108002701280,170108002701388,170109002101012,170109003801078,170109003801118,170109004301307,170112001601321,170112003601099,170112003601141,170112003601380,170113002101327,170114004701208,170114005201043,170114005201045,170114005801116,170115004701077,170115005201123,170115005801016,170118002701149,170119003101377,170121002201016,170121002201041,170121002801218,170121002801327,170121003401109,170121004501388,170122002101256,170122002601354,170128001601394,170130001601011,170130002601232,170130003101338,170130003601395,170206004201039,170217001601271,170218003701083,170219002101073,170219002101284,170220001601123,170220004101184,170220004601155,170312001601076,170403001601117,170404001601032,170404002601278,170407004101259,170408004001351,170408005501185,170410003901140,170410004501209,170411004601097,170413003101109,170413003101119,170413006101068,170414003101193,170414003601184,170415001501109,170415001501267,170415001501314,170415002501109,170415002501129,170415002501267,170415002501314,170416001901081,170416001901090,170416001901104,170416001901112,170416001901119,170416001901167,170416003801037,170416004801238,170418001601032,170418002101087,170418002101143,170506003901216,170506005401133,170506005401242,170506005401271,170506005401391,170506005901185,170506006401020,170506006401204,170506006401219,170506006401238,170506006401240,170506006401243,170506006401365,170506006401376,170507010601384,170507010601395,170508002601106,170508003801315,170508004801243,170509002201359,170509002701247,170509003701234,170509003701284,170509004701233,170509004701262,170509005701310,170509005701368,170509007701254,170509007701279,170509007701280,170509007701282,170509007701291,170509007701292,170509007701296,170510001801246,170510001801342,170510005801246,170510005801258,170510006301386,170510007801149,170510007801171,170510007801237,170510007801260,170510007801284,170511000101123,170511000101247,170511000101342,170511003301310,170511003301347,170511004001056,170513003501228,170513004901063,170513005401121,170513005901184,170513005901238,170513005901327,170514001601104,170514001601166,170514003301345,170514003801275,170515003101298,170515004601154,170515006101008,170515006101145,170516002101075,170517001801021,170517001801136,170601001601184,170601001601185,170601001601214,170601001601221,170601001601253,170601001601288,170601001601296,170601001601315,170601001601325,170601001601333,170601001601334,170601002601104,170614005101145,170615002801320,170615003401321,170615004401216,170615004401260,170615004901298,170710002201393,170713001601243,170713002101395,170724001601311,170828001101028,170828001101029,170828001101077,170828001101079,170828001101080,170828001101138,170828001101156,170828001101241,170828001101246,170828001101258,170828001101273,170828001101348,170828001101365,170828001101375,170828001601015,170828001601019,170828001601028,170828001601032,170828001601080,170828001601087,170828001601141,170828001601156,170828001601167,170828001601241,170828001601246,170828001601257,170828001601272,170828001601338,170828001601375,170828001601396,170828002201015,170828002201038,170828002201047,170828002201053,170828002201055,170828002201056,170828002201076,170828002201077,170828002201083,170828002201094,170828002201097,170828002201098,170828002201099,170828002201102,170828002201104,170828002201105,170828002201109,170828002201113,170828002201124,170828002201125,170828002201131,170828002201141,170828002201142,170828002201146,170828002201149,170828002201158,170828002201172,170828002201178,170828002201186,170828002201190,170828002201192,170828002201195,170828002201201,170828002201208,170828002201209,170828002201210,170828002201220,170828002201224,170828002201225,170828002201235,170828002201237,170828002201246,170828002201247,170828002201255,170828002201258,170828002201263,170828002201266,170828002201281,170828002201282,170828002201285,170828002201289,170828002201291,170828002201299,170828002201308,170828002201312,170828002201313,170828002201314,170828002201327,170828002201343,170828002201344,170828002201354,170828002201381,170828002201392,170828002701019,170828002701095,170828002701097,170828002701137,170828002701164,170828002701168,170828002701170,170828002701204,170828002701287,170828002701296,170828002701394,170828003201009,170828003201015,170828003201023,170828003201034,170828003201048,170828003201053,170828003201070,170828003201078,170828003201080,170828003201089,170828003201153,170828003201156,170828003201161,170828003201191,170828003201211,170828003201224,170828003201240,170828003201255,170828003201285,170828003201295,170828003201355,170828003201384,170828003201390,170828003201394,170828003901005,170828003901022,170828003901086,170828003901129,170828003901187,170828003901237,170828003901251,170828003901253,170828003901291,170828003901334,170828003901384,170828004401029,170828004401068,170828004401178,170828004401279,170828004401325,170828004401341,170829001101003,170829001101006,170829001101008,170829001101009,170829001101029,170829001101031,170829001101035,170829001101065,170829001101080,170829001101081,170829001101092,170829001101102,170829001101127,170829001101142,170829001101146,170829001101147,170829001101154,170829001101157,170829001101171,170829001101176,170829001101201,170829001101242,170829001101251,170829001101255,170829001101269,170829001101311,170829001101314,170829001101378,170829001101387,170829001101392,170829001901071,170829001901085,170829001901159,170829001901165,170829001901178,170829001901240,170829001901241,170829001901274,170829001901341,170829001901343,170829001901348,170829001901392,170829002401026,170829002401071,170829002401158,170829002401165,170829002401177,170829002401241,170829002401341,170829002401381,170829002401392,170829002401395,170829003901072,170829003901184,170829003901358,170830001101242,170830001101254,170830001101299,170830001101392,170830001801317,170830002301015,170830002301097,170830002301128,170830002301142,170830002301172,170830002301224,170830002301299,170830002301343,170830002301381,170830002301392,170830002801156,170830002801346,170830002801394,170830003401394,170830004001131,170830004801072,170830004801184,170830004801358,170912001201012,170912001201033,170912001201056,170912001201075,170912001201087,170912001201151,170912001201165,170912001201216,170912001201267,170912001201389,170912001901068,170912001901114,171027003801040,171027003801246,171029004301246,171029004301384,171031004101053,171031004101386,171031004101389,171031004501055,171031004501379,171228001601335,171228001601342,171228001601353,171228001601357,171228001601374,171230003101119,171230003101120,171230003101123,171230003101149,180102001601234,180102001601241,180125001901066,180125001901262,180125002501030,180125002501074,180125002501101,180125002501105,180125002501106,180125002501114,180125002501348,180126001601066,180126001601112,180126001601196,180126001601302,180126002101043,180126002101054,180126002101065,180126002101080,180126002601022,180126002601112,180126002601198,180126002601276,180126002601308,180126002601325,180131002701374]
abund_param = unique(vstack([cluster_param, abund_param]), keys=['sobject_id'])
abund_param = abund_param[np.in1d(abund_param['sobject_id'], remove, invert=True)]
idx_bad = np.logical_or(abund_param['flag_guess'] != 0, abund_param['red_flag'] != 0)
# abund_param = abund_param[~idx_bad]
idx_gc = np.in1d(abund_param['sobject_id'], globc_param['sobject_id'])
print len(abund_param[~idx_bad])

out_dir = 'Train_plots'
os.system('mkdir '+out_dir)
os.chdir(out_dir)

# select only the ones with some datapoints
sme_abundances_list = [col for col in sme_abundances_list if np.sum(np.isfinite(abund_param[col])) > 100]
sme_abundances_list = [col for col in sme_abundances_list if len(col.split('_')[0]) <= 3]
print 'SME Abundances:', sme_abundances_list
print 'Number Abund:', len(sme_abundances_list)

# H-R diagnostics plot
fig, ax = plt.subplots(1, 1)
ax.scatter(abund_param['teff'], abund_param['logg'], s=2, alpha=1., lw=0, c='black', label='SME')
ax.scatter(abund_param['teff'][idx_bad], abund_param['logg'][idx_bad], s=2, alpha=1., lw=0, c='blue', label='Red/guess flag')
ax.scatter(abund_param['teff'][idx_gc], abund_param['logg'][idx_gc], s=2, alpha=1., lw=0, c='green', label='GC')
ax.set(xlabel='Teff', ylabel='logg', xlim=(7600, 3300), ylim=(5.5, 0))
ax.legend()
selector = PointSelector(ax, 'teff', 'logg')
# plt.show()
plt.savefig('kiel.png', dpi=250)
plt.close()

for abund in sme_abundances_list:
    idx_flag = abund_param['flag_A_'+abund.split('_')[0]] > 0
    print abund, np.sum(idx_flag)
    # H-R diagnostics plot
    fig, ax = plt.subplots(1, 1)
    ax.scatter(abund_param['fe_h'], abund_param[abund], s=2, alpha=1., lw=0, c='black', label='SME')
    ax.scatter(abund_param['fe_h'][idx_bad], abund_param[abund][idx_bad], s=2, alpha=1., lw=0, c='blue', label='Red/guess flag')
    ax.scatter(abund_param['fe_h'][idx_gc], abund_param[abund][idx_gc], s=2, alpha=1., lw=0, c='green', label='GC')
    ax.scatter(abund_param['fe_h'][idx_flag], abund_param[abund][idx_flag], s=2, alpha=1., lw=0, c='red', label='flag')
    ax.set(xlabel='Fe/H', ylabel=abund, xlim=(-3, 1), ylim=(-2, 2))
    ax.legend()
    selector = PointSelector(ax, 'fe_h', abund)
    # plt.show()
    plt.savefig(abund+'.png', dpi=250)
    plt.close()

outliers.append(remove)
bad_sid = np.unique(np.hstack(outliers))
print ','.join([str(s) for s in bad_sid])
