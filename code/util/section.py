import os
import os.path
import re

# Explorative approach 1: Writing templates in attempt to match ~30%
# Explorative approach 2: Match possible templates, hand tweak, and divide
#  (gets everything segmented, but may not do it correctly)
# TODO: Explorative approach 3: Leverage possible section count -> Prob(section)

POSSIBLE_SECTIONS = 'sections_hand.txt'
def sectionify(note, sections=None):
    if not sections:
        with open(POSSIBLE_SECTIONS) as f:
            sheads = [s.strip() for s in f]
    regex = "(" + "|".join(sheads) + ")"
    regex = regex.tolower()
    snote = re.split(regex, note.tolower())
    return snote


# Approach 1: Accurate labeling, but insufficient
REGEX_FLAGS = re.MULTILINE | re.VERBOSE | re.DOTALL
NURSE_NOTES_PATH = '/scratch/tjn/nursenotes26_filtered'
TEMPLATES = [
r"""    # modeled from NOTE-EVENTS-10000.txt
\s*
^Admission[ ]Date: (.*?)	Discharge[ ]Date: (.*?)
^Date[ ]of[ ]Birth: (.*?)	Sex: (.*?)
^Service: (.*?)
^Allergies: (.*?)
^Attending: (.*?)
^Chief[ ]Complaint: (.*?)
^Major[ ]Surgical[ ]or[ ]Invasive[ ]Procedure: (.*?)
^History[ ]of[ ]Present[ ]Illness: (.*?)
^Past[ ]Medical[ ]History: (.*?)
^Social[ ]History: (.*?)
^Family[ ]History: (.*?)
^Physical[ ]Exam: (.*?)
^Pertinent[ ]Results: (.*?)
^Brief[ ]Hospital[ ]Course: (.*?)
^Medications[ ]on[ ]Admission: (.*?)
^Discharge[ ]Medications: (.*?)
^Discharge[ ]Disposition: (.*?)
^Facility: (.*?)
^Discharge[ ]Diagnosis: (.*?)
^Discharge[ ]Condition: (.*?)
^Discharge[ ]Instructions: (.*?)
^Followup[ ]Instructions: (.*?)
^(\s* (?:Dr[.]\s?)? [[].*?)
^\(End[ ]of[ ]Report\)
""",

r"""	# modeled from NOTE-EVENTS-00106.txt
\s*
^Admission[ ]Date: (.*?)	Discharge[ ]Date: (.*?)
^Date[ ]of[ ]Birth: (.*?)	Sex: (.*?)
^Service: (.*?)
^Allergies: (.*?)
^Attending: (.*?)
^Chief[ ]Complaint: (.*?)
^Major[ ]Surgical[ ]or[ ]Invasive[ ]Procedure: (.*?)
^History[ ]of[ ]Present[ ]Illness: (.*?)
^Past[ ]Medical[ ]History: (.*?)
^Social[ ]History: (.*?)
^Family[ ]History: (.*?)
^Physical[ ]Exam: (.*?)
^Pertinent[ ]Results: (.*?)
^Brief[ ]Hospital[ ]Course: (.*?)
^Medications[ ]on[ ]Admission: (.*?)
^Discharge[ ]Medications: (.*?)
^Discharge[ ]Disposition: (.*?)
^Discharge[ ]Diagnosis: (.*?)
^Discharge[ ]Condition: (.*?)
^Discharge[ ]Instructions: (.*?)
^Followup[ ]Instructions: (.*?)
^(\s* (?:Dr[.]\s?)? [[].*?)
^\(End[ ]of[ ]Report\)
""",

r"""	# modeled from NOTE-EVENTS-00097.txt
\s*
^Admission[ ]Date: (.*?)	Discharge[ ]Date: (.*?)
^Date[ ]of[ ]Birth: (.*?)	Sex: (.*?)
^Service: (.*?)
^HISTORY[ ]OF[ ]PRESENT[ ]ILLNESS: (.*?)
^PAST[ ]MEDICAL[ ]HISTORY: (.*?)
^MEDICATIONS[ ]ON[ ]ADMISSION: (.*?)
^ALLERGIES: (.*?)
^FAMILY[ ]HISTORY: (.*?)
^SOCIAL[ ]HISTORY: (.*?)
^PHYSICAL[ ]EXAMINATION: (.*?)
^LABORATORY[ ]DATA: (.*?)
^ASSESSMENT[ ]AND[ ]PLAN: (.*?)
^HOSPITAL[ ]COURSE: (.*?)
^DISCHARGE[ ]PLANNING: (.*?)
^DISCHARGE[ ]MEDICATIONS: (.*?)
^DISCHARGE[ ]DIAGNOSIS: (.*?)
^DISCHARGE[ ]STATUS: (.*?)
^(\s* (?:Dr[.]\s?)? [[].*?)
^\(End[ ]of[ ]Report\)
""",

r"""	# modeled from NOTE-EVENTS-00067.txt
\s*
^Admission[ ]Date: (.*?)	Discharge[ ]Date: (.*?)
^Date[ ]of[ ]Birth: (.*?)	Sex: (.*?)
^Service: (.*?)
^HISTORY[ ]OF[ ]PRESENT[ ]ILLNESS: (.*?)
^PAST[ ]MEDICAL[ ]HISTORY: (.*?)
^PAST[ ]SURGICAL[ ]HISTORY: (.*?)
^MEDICATIONS: (.*?)
^ALLERGIES: (.*?)
^LABORATORIES[ ]ON[ ]ADMISSION: (.*?)
^RADIOLOGIC[ ]EXAMINATION[ ]ON[ ]ADMISSION: (.*?)
^ASSESSMENT[ ]AND[ ]PLAN: (.*?)
^BRIEF[ ]HOSPITAL[ ]COURSE: (.*?)
^(\s* (?:Dr[.]\s?)? [[].*?)
^\(End[ ]of[ ]Report\)
""",

r"""	# modeled from NOTE-EVENTS-00113.txt
\s*
^Admission[ ]Date: (.*?)	Discharge[ ]Date: (.*?)
^Date[ ]of[ ]Birth: (.*?)	Sex: (.*?)
^Service: (.*?)
^HISTORY[ ]OF[ ]PRESENT[ ]ILLNESS: (.*?)
^PAST[ ]SURGICAL[ ]HISTORY: (.*?)
^PAST[ ]MEDICAL[ ]HISTORY: (.*?)
^PHYSICAL[ ]EXAMINATION: (.*?)
^HOSPITAL[ ]COURSE: (.*?)
^DISPOSITION: (.*?)
^(\s* (?:Dr[.]\s?)? [[].*?)
^\(End[ ]of[ ]Report\)
""",

r"""	# modeled from NOTE-EVENTS-00080.txt
\s*
^Admission[ ]Date: (.*?)	Discharge[ ]Date: (.*?)
^Date[ ]of[ ]Birth: (.*?)	Sex: (.*?)
^Service: (.*?)
^HISTORY[ ]OF[ ]PRESENT[ ]ILLNESS: (.*?)
^PAST[ ]MEDICAL[ ]HISTORY: (.*?)
^PAST[ ]SURGICAL[ ]HISTORY: (.*?)
^ALLERGIES: (.*?)
^MEDICATIONS: (.*?)
^SOCIAL[ ]HISTORY: (.*?)
^PHYSICAL[ ]EXAMINATION: (.*?)
^LABORATORY[ ]DATA: (.*?)
^HOSPITAL[ ]COURSE: (.*?)
^CONDITION[ ]ON[ ]DISCHARGE: (.*?)
^DISCHARGE[ ]STATUS: (.*?)
^DISCHARGE[ ]DIAGNOSES: (.*?)
^(\s* (?:Dr[.]\s?)? [[].*?)
^\(End[ ]of[ ]Report\)
""",

r"""	# modeled from NOTE-EVENTS-00113.txt
\s*
^Admission[ ]Date: (.*?)	Discharge[ ]Date: (.*?)
^Date[ ]of[ ]Birth: (.*?)	Sex: (.*?)
^Service: (.*?)
^HOSPITAL[ ]COURSE: (.*?)
^CONDITION[ ]ON[ ]DISCHARGE: (.*?)
^DISCHARGE[ ]DISPOSITION: (.*?)
^PRIMARY[ ]PEDIATRICIAN: (.*?)
^CARE/RECOMMENDATIONS: (.*?)
^DISCHARGE[ ]DIAGNOSIS: (.*?)
^(\s* (?:Dr[.]\s?)? [[].*?)
^\(End[ ]of[ ]Report\)
""",

r"""	# modeled from NOTE-EVENTS-00108.txt
\s*
^Admission[ ]Date: (.*?)	Discharge[ ]Date: (.*?)
^Date[ ]of[ ]Birth: (.*?)	Sex: (.*?)
^Service: (.*?)
^HISTORY[ ]OF[ ]PRESENT[ ]ILLNESS: (.*?)
^PAST[ ]MEDICAL[ ]HISTORY: (.*?)
^MEDICATIONS[ ]ON[ ]ADMISSION: (.*?)
^HOSPITAL[ ]COURSE: (.*?)
^DISPOSITION: (.*?)
^DISCHARGE[ ]MEDICATIONS: (.*?)
^FOLLOW[ ]UP: (.*?)
^(\s* (?:Dr[.]\s?)? [[].*?)
^\(End[ ]of[ ]Report\)
""",

r"""	# modeled from NOTE-EVENTS-00033.txt
\s*
^Admission[ ]Date: (.*?)	Discharge[ ]Date: (.*?)
^Date[ ]of[ ]Birth: (.*?)	Sex: (.*?)
^Service: (.*?)
^HISTORY[ ]OF[ ]PRESENT[ ]ILLNESS: (.*?)
^REVIEW[ ]OF[ ]SYSTEMS: (.*?)
^PHYSICAL[ ]EXAM[ ]ON[ ]ADMISSION: (.*?)
^LABORATORIES[ ]ON[ ]ADMISSION: (.*?)
^BRIEF[ ]SUMMARY[ ]OF[ ]HOSPITAL[ ]COURSE: (.*?)
^DISCHARGE[ ]DIAGNOSES: (.*?)
^FOLLOW-UP[ ]INSTRUCTIONS: (.*?)
^DISCHARGE[ ]CONDITION: (.*?)
^DISCHARGE[ ]MEDICATIONS: (.*?)
^(\s* (?:Dr[.]\s?)? [[].*?)
^\(End[ ]of[ ]Report\)
""",

r"""	# modeled from NOTE-EVENTS-00124.txt
\s*
^Admission[ ]Date: (.*?)	Discharge[ ]Date: (.*?)
^Date[ ]of[ ]Birth: (.*?)	Sex: (.*?)
^Service: (.*?)
^HISTORY[ ]OF[ ]PRESENT[ ]ILLNESS: (.*?)
^MEDICATIONS[ ]PRIOR[ ]TO[ ]ADMISSION: (.*?)
^PAST[ ]MEDICAL[ ]HISTORY: (.*?)
^SOCIAL[ ]HISTORY: (.*?)
^PHYSICAL[ ]EXAMINATION: (.*?)
^LABORATORY[ ]ON[ ]ADMISSION: (.*?)
^HOSPITAL[ ]COURSE: (.*?)
^DISCHARGE[ ]MEDICATIONS: (.*?)
^DISCHARGE[ ]INSTRUCTIONS: (.*?)
^(\s* (?:Dr[.]\s?)? [[].*?)
^\(End[ ]of[ ]Report\)
""",

r"""	# modeled from NOTE-EVENTS-00087.txt
\s*
^Admission[ ]Date: (.*?)	Discharge[ ]Date: (.*?)
^Date[ ]of[ ]Birth: (.*?)	Sex: (.*?)
^Service: (.*?)
^IDENTIFCATION: (.*?)
^HISTORY: (.*?)
^EXAMINATION[ ]ON[ ]ADMISSION: (.*?)
^SUMMARY[ ]OF[ ]HOSPITAL[ ]COURSE[ ]BY[ ]SYSTEMS: (.*?)
^CONDITION[ ]AT[ ]DISCHARGE: (.*?)
^DISCHARGE[ ]DISPOSITION: (.*?)
^PRIMARY[ ]PEDIATRICIAN: (.*?)
^CARE[ ]RECOMMENDATIONS: (.*?)
^DISCHARGE[ ]DIAGNOSES: (.*?)
^(\s* (?:Dr[.]\s?)? [[].*?)
^\(End[ ]of[ ]Report\)
""",

r"""	# modeled from NOTE-EVENTS-00083.txt
\s*
^Admission[ ]Date: (.*?)	Discharge[ ]Date: (.*?)
^Date[ ]of[ ]Birth: (.*?)	Sex: (.*?)
^Service: (.*?)
^HISTORY[ ]OF[ ]THE[ ]PRESENT[ ]ILLNESS: (.*?)
^REVIEW[ ]OF[ ]SYSTEMS: (.*?)
^PAST[ ]MEDICAL[ ]HISTORY: (.*?)
^ALLERGIES: (.*?)
^ADMISSION[ ]MEDICATIONS: (.*?)
^SOCIAL[ ]HISTORY: (.*?)
^PHYSICAL[ ]EXAMINATION[ ]ON[ ]ADMISSION: (.*?)
^ADMISSION[ ]LABORATORY[ ]DATA/STUDIES: (.*?)
^HOSPITAL[ ]COURSE: (.*?)
^DISCHARGE[ ]DIAGNOSIS: (.*?)
^DISPOSITION: (.*?)
^DISCHARGE[ ]MEDICATIONS: (.*?)
^FOLLOW-UP: (.*?)
^(\s* (?:Dr[.]\s?)? [[].*?)
^\(End[ ]of[ ]Report\)
""",

r"""	# modeled from NOTE-EVENTS-00078.txt
\s*
^Admission[ ]Date: (.*?)	Discharge[ ]Date: (.*?)
^Date[ ]of[ ]Birth: (.*?)	Sex: (.*?)
^Service: (.*?)
^CHIEF[ ]COMPLAINT: (.*?)
^HISTORY[ ]OF[ ]PRESENT[ ]ILLNESS: (.*?)
^PAST[ ]MEDICAL[ ]HISTORY: (.*?)
^MEDICATIONS[ ]PRIOR[ ]TO[ ]ADMISSION: (.*?)
^SOCIAL[ ]HISTORY: (.*?)
^ALLERGIES: (.*?)
^FAMILY[ ]HISTORY: (.*?)
^REVIEW[ ]OF[ ]SYSTEMS: (.*?)
^PHYSICAL[ ]EXAMINATION[ ]AT[ ]PRESENTATION: (.*?)
^PERTINENT[ ]LABORATORY[ ]DATA[ ]ON[ ]PRESENTATION: (.*?)
^RADIOLOGY\/STUDIES[ ]AT[ ]ADMISSION: (.*?)
^IMPRESSION[ ]AT[ ]ADMISSION: (.*?)
^HOSPITAL[ ]COURSE[ ]IN[ ]THE[ ]INTENSIVE[ ]CARE[ ]UNIT: (.*?)
^SUMMARY[ ]OF[ ]INTENSIVE[ ]CARE[ ]UNIT[ ]STAY: (.*?)
^HOSPITAL[ ]COURSE[ ]ON[ ]THE[ ]FLOOR[ ]\(\[\*\*2778\-3\-25\*\*\][ ]to[ ]\[\*\*2778\-3\-27\*\*\]\): (.*?)
^CONDITION[ ]AT[ ]DISCHARGE: (.*?)
^DISCHARGE[ ]STATUS: (.*?)
^DISCHARGE[ ]MEDICATIONS: (.*?)
^DISCHARGE[ ]INSTRUCTIONS[ ]AND[ ]FOLLOW[ ]UP: (.*?)
^DISCHARGE[ ]DIAGNOSES: (.*?)
^(\s* (?:Dr[.]\s?)? [[].*?)
^\(End[ ]of[ ]Report\)
""",

r"""	# modeled from NOTE-EVENTS-00065.txt
\s*
^Admission[ ]Date: (.*?)	Discharge[ ]Date: (.*?)
^Date[ ]of[ ]Birth: (.*?)	Sex: (.*?)
^Service: (.*?)
^ADDENDUM: (.*?)
^MEDICATIONS[ ]ON[ ]DISCHARGE: (.*?)
^DISCHARGE[ ]STATUS: (.*?)
^(\s* (?:Dr[.]\s?)? [[].*?)
^\(End[ ]of[ ]Report\)
""",

r"""	# modeled from NOTE-EVENTS-00056.txt
\s*
^Admission[ ]Date: (.*?)	Discharge[ ]Date: (.*?)
^Date[ ]of[ ]Birth: (.*?)	Sex: (.*?)
^Service: (.*?)
^Allergies: (.*?)
^Attending: (.*?)
^Major[ ]Surgical[ ]or[ ]Invasive[ ]Procedure: (.*?)
^History[ ]of[ ]Present[ ]Illness: (.*?)
^Past[ ]Medical[ ]History: (.*?)
^Social[ ]History: (.*?)
^Physical[ ]Exam: (.*?)
^Pertinent[ ]Results: (.*?)
^Brief[ ]Hospital[ ]Course: (.*?)
^Medications[ ]on[ ]Admission: (.*?)
^Discharge[ ]Medications: (.*?)
^Discharge[ ]Disposition: (.*?)
^Facility: (.*?)
^Discharge[ ]Diagnosis: (.*?)
^Discharge[ ]Condition: (.*?)
^Discharge[ ]Instructions: (.*?)
^Followup[ ]Instructions: (.*?)
^(\s* (?:Dr[.]\s?)? [[].*?)
^\(End[ ]of[ ]Report\)
""",

r"""	# modeled from NOTE-EVENTS-00026.txt
\s*
^Admission[ ]Date: (.*?)	Discharge[ ]Date: (.*?)
^Date[ ]of[ ]Birth: (.*?)	Sex: (.*?)
^Service: (.*?)
^HISTORY[ ]OF[ ]PRESENT[ ]ILLNESS: (.*?)
^PAST[ ]MEDICAL[ ]HISTORY: (.*?)
^MEDICATIONS: (.*?)
^SOCIAL[ ]HISTORY: (.*?)
^ALLERGIES: (.*?)
^PHYSICAL[ ]EXAM[ ]ON[ ]ADMISSION: (.*?)
^SUMMARY[ ]OF[ ]HOSPITAL[ ]COURSE: (.*?)
^CONDITION[ ]ON[ ]DISCHARGE: (.*?)
^DISCHARGE[ ]STATUS: (.*?)
^DISCHARGE[ ]DIAGNOSES: (.*?)
^DISCHARGE[ ]MEDICATIONS: (.*?)
^FOLLOW\-UP[ ]PLANS: (.*?)
^(\s* (?:Dr[.]\s?)? [[].*?)
^\(End[ ]of[ ]Report\)
""",

r"""	# modeled from NOTE-EVENTS-00040.txt
\s*
^Admission[ ]Date: (.*?)	Discharge[ ]Date: (.*?)
^Date[ ]of[ ]Birth: (.*?)	Sex: (.*?)
^Service: (.*?)
^HISTORY[ ]OF[ ]PRESENT[ ]ILLNESS: (.*?)
^PAST[ ]MEDICAL[ ]HISTORY: (.*?)
^ALLERGIES: (.*?)
^MEDICATIONS[ ]AT[ ]HOME: (.*?)
^MEDICATIONS[ ]ON[ ]TRANSFER: (.*?)
^SOCIAL[ ]HISTORY: (.*?)
^PHYSICAL[ ]EXAMINATION[ ]ON[ ]ADMISSION: (.*?)
^LABORATORY[ ]VALUES[ ]ON[ ]ADMISSION: (.*?)
^SUMMARY[ ]OF[ ]HOSPITAL[ ]COURSE: (.*?)
^(\s* (?:Dr[.]\s?)? [[].*?)
^\(End[ ]of[ ]Report\)
""",

r"""	# modeled from NOTE-EVENTS-00102.txt
\s*
^Admission[ ]Date: (.*?)	Discharge[ ]Date: (.*?)
^Date[ ]of[ ]Birth: (.*?)	Sex: (.*?)
^Service: (.*?)
^DISCHARGE[ ]DIAGNOSES: (.*?)
^HISTORY[ ]OF[ ]PRESENT[ ]ILLNESS: (.*?)
^PROBLEMS[ ]DURING[ ]HOSPITAL[ ]STAY: (.*?)
^DISCHARGE[ ]MEDICATIONS: (.*?)
^FOLLOW\-UP[ ]PLAN: (.*?)
^(\s* Dictated[ ]By:[[].*?)
^\(End[ ]of[ ]Report\)
""",

r"""	# modeled from NOTE-EVENTS-00069.txt
\s*
^Admission[ ]Date: (.*?)	Discharge[ ]Date: (.*?)
^Date[ ]of[ ]Birth: (.*?)	Sex: (.*?)
^Service: (.*?)
^HISTORY: (.*?)
^ANTENATAL[ ]HISTORY: (.*?)
^NEONATAL[ ]COURSE: (.*?)
^SUMMARY[ ]OF[ ]HOSPITAL[ ]COURSE[ ]BY[ ]SYSTEMS: (.*?)
^NAME[ ]OF[ ]PRIMARY[ ]PEDIATRICIAN: (.*?)
^DISCHARGE[ ]DIAGNOSIS: (.*?)
^(\s* (?:Dr[.]\s?)? [[].*?)
^\(End[ ]of[ ]Report\)
""",

r"""	# modeled from NOTE-EVENTS-00015.txt
\s*
^Admission[ ]Date: (.*?)	Discharge[ ]Date: (.*?)
^Date[ ]of[ ]Birth: (.*?)	Sex: (.*?)
^Service: (.*?)
^HISTORY[ ]OF[ ]PRESENT[ ]ILLNESS: (.*?)
^PAST[ ]PSYCHIATRIC[ ]HISTORY: (.*?)
^PAST[ ]MEDICAL[ ]HISTORY: (.*?)
^ALLERGIES: (.*?)
^MEDICATIONS[ ]ON[ ]ADMISSION: (.*?)
^SUBSTANCE[ ]ABUSE[ ]HISTORY: (.*?)
^SOCIAL[ ]HISTORY: (.*?)
^FAMILY[ ]PSYCHIATRIC[ ]HISTORY: (.*?)
^LABORATORY[ ]DATA: (.*?)
^MENTAL[ ]STATUS[ ]EXAM: (.*?)
^HOSPITAL[ ]COURSE: (.*?)
^CONDITION[ ]ON[ ]DISCHARGE: (.*?)
^DISCHARGE[ ]DIAGNOSIS: (.*?)
^DISCHARGE[ ]MEDICATIONS: (.*?)
^FOLLOW\-UP: (.*?)
^(\s* (?:Dr[.]\s?)? [[].*?)
^\(End[ ]of[ ]Report\)
""",

r"""	# modeled from NOTE-EVENTS-00011.txt
\s*
^Admission[ ]Date: (.*?)	Discharge[ ]Date: (.*?)
^Date[ ]of[ ]Birth: (.*?)	Sex: (.*?)
^Service: (.*?)
^Allergies: (.*?)
^Attending: (.*?)
^Chief[ ]Complaint: (.*?)
^Major[ ]Surgical[ ]or[ ]Invasive[ ]Procedure: (.*?)
^History[ ]of[ ]Present[ ]Illness: (.*?)
^Past[ ]Medical[ ]History: (.*?)
^Family[ ]History: (.*?)
^Physical[ ]Exam: (.*?)
^Pertinent[ ]Results: (.*?)
^Brief[ ]Hospital[ ]Course: (.*?)
^Medications[ ]on[ ]Admission: (.*?)
^Discharge[ ]Medications: (.*?)
^Discharge[ ]Disposition: (.*?)
^Discharge[ ]Diagnosis: (.*?)
^Discharge[ ]Condition: (.*?)
^Discharge[ ]Instructions: (.*?)
^Followup[ ]Instructions: (.*?)
^(\s* (?:Dr[.]\s?)? [[].*?)
^\(End[ ]of[ ]Report\)
""",

r"""	# modeled from NOTE-EVENTS-00053.txt
\s*
^Admission[ ]Date: (.*?)	Discharge[ ]Date: (.*?)
^Date[ ]of[ ]Birth: (.*?)	Sex: (.*?)
^Service: (.*?)
^HISTORY[ ]OF[ ]PRESENT[ ]ILLNESS: (.*?)
^PHYSICAL[ ]EXAMINATION: (.*?)
^LABORATORY[ ]DATA: (.*?)
^HOSPITAL[ ]COURSE: (.*?)
^DISCHARGE[ ]STATUS: (.*?)
^DISCHARGE[ ]DISPOSITION: (.*?)
^FOLLOWUP[ ]PLANS: (.*?)
^DISCHARGE[ ]MEDICATIONS: (.*?)
^(\s* (?:Dr[.]\s?)? [[].*?)
^\(End[ ]of[ ]Report\)
""",

r"""	# modeled from NOTE-EVENTS-00012.txt
\s*
^Admission[ ]Date: (.*?)	Discharge[ ]Date: (.*?)
^Date[ ]of[ ]Birth: (.*?)	Sex: (.*?)
^Service: (.*?)
^HISTORY[ ]OF[ ]PRESENT[ ]ILLNESS: (.*?)
^PAST[ ]MEDICAL[ ]HISTORY: (.*?)
^PAST[ ]SURGICAL[ ]HISTORY: (.*?)
^MEDICATIONS[ ]ON[ ]ADMISSION: (.*?)
^ALLERGIES: (.*?)
^PHYSICAL[ ]EXAMINATION[ ]ON[ ]PRESENTATION: (.*?)
^PERTINENT[ ]LABORATORY[ ]VALUES[ ]ON[ ]PRESENTATION: (.*?)
^BRIEF[ ]SUMMARY[ ]OF[ ]HOSPITAL[ ]COURSE: (.*?)
^CONDITION[ ]AT[ ]DISCHARGE: (.*?)
^(\s* (?:Dr[.]\s?)? [[].*?)
^\(End[ ]of[ ]Report\)
""",

r"""	# modeled from NOTE-EVENTS-00073.txt
\s*
^Admission[ ]Date: (.*?)	Discharge[ ]Date: (.*?)
^Date[ ]of[ ]Birth: (.*?)	Sex: (.*?)
^Service: (.*?)
^CHIEF[ ]COMPLAINT: (.*?)
^HISTORY[ ]OF[ ]PRESENT[ ]ILLNESS: (.*?)
^PAST[ ]MEDICAL[ ]HISTORY: (.*?)
^MEDICATIONS[ ]ON[ ]ADMISSION: (.*?)
^ALLERGIES: (.*?)
^SOCIAL[ ]HISTORY: (.*?)
^FAMILY[ ]HISTORY: (.*?)
^PHYSICAL[ ]EXAMINATION[ ]ON[ ]PRESENTATION: (.*?)
^PERTINENT[ ]LABORATORY[ ]DATA[ ]ON[ ]PRESENTATION: (.*?)
^RADIOLOGY\/IMAGING: (.*?)
^HOSPITAL[ ]COURSE: (.*?)
^CONDITION[ ]AT[ ]DISCHARGE: (.*?)
^DISCHARGE[ ]STATUS: (.*?)
^DISCHARGE[ ]DIAGNOSES: (.*?)
^MEDICATIONS[ ]ON[ ]DISCHARGE: (.*?)
^DISCHARGE[ ]INSTRUCTIONS\/FOLLOWUP: (.*?)
^(\s* (?:Dr[.]\s?)? [[].*?)
^\(End[ ]of[ ]Report\)
""",

r"""	# modeled from NOTE-EVENTS-00089.txt
\s*
^Admission[ ]Date: (.*?)	Discharge[ ]Date: (.*?)
^Date[ ]of[ ]Birth: (.*?)	Sex: (.*?)
^Service: (.*?)
^HISTORY: (.*?)
^HOSPITAL[ ]COURSE[ ]BY[ ]SYSTEMS: (.*?)
^CONDITION[ ]ON[ ]DISCHARGE: (.*?)
^CARE[ ]AND[ ]RECOMMENDATIONS[ ]AFTER[ ]DISCHARGE: (.*?)
^MEDICATIONS: (.*?)
^DISCHARGE[ ]DIAGNOSES: (.*?)
^(\s* (?:Dr[.]\s?)? [[].*?)
^\(End[ ]of[ ]Report\)
""",

r"""	# modeled from NOTE-EVENTS-00110.txt
\s*
^Admission[ ]Date: (.*?)	Discharge[ ]Date: (.*?)
^Date[ ]of[ ]Birth: (.*?)	Sex: (.*?)
^Service: (.*?)
^ID: (.*?)
^HISTORY: (.*?)
^PHYSICAL[ ]EXAMINATION[ ]ON[ ]ADMISSION: (.*?)
^HOSPITAL[ ]COURSE: (.*?)
^CONDITION[ ]ON[ ]DISCHARGE: (.*?)
^DISPOSITION: (.*?)
^PRIMARY[ ]PEDIATRICIAN: (.*?)
^CARE[ ]RECOMMENDATIONS: (.*?)
^DISCHARGE[ ]DIAGNOSES: (.*?)
^(\s* (?:Dr[.]\s?)? [[].*?)
^\(End[ ]of[ ]Report\)
""",

r"""	# modeled from NOTE-EVENTS-00123.txt
\s*
^Admission[ ]Date: (.*?)	Discharge[ ]Date: (.*?)
^Date[ ]of[ ]Birth: (.*?)	Sex: (.*?)
^Service: (.*?)
^HISTORY[ ]OF[ ]PRESENT[ ]ILLNESS: (.*?)
^PAST[ ]MEDICAL[ ]HISTORY: (.*?)
^ALLERGIES: (.*?)
^MEDICATIONS[ ]ON[ ]ADMISSION: (.*?)
^FAMILY[ ]HISTORY: (.*?)
^SOCIAL[ ]HISTORY: (.*?)
^PHYSICAL[ ]EXAMINATION[ ]ON[ ]PRESENTATION: (.*?)
^PERTINENT[ ]LABORATORY[ ]DATA[ ]ON[ ]PRESENTATION: (.*?)
^HOSPITAL[ ]COURSE[ ]BY[ ]PROBLEM: (.*?)
^DISCHARGE[ ]FOLLOWUP: (.*?)
^DISCHARGE[ ]DIAGNOSES: (.*?)
^MEDICATIONS[ ]ON[ ]DISCHARGE: (.*?)
^(\s* (?:Dr[.]\s?)? [[].*?)
^\(End[ ]of[ ]Report\)
""",

r"""	# modeled from NOTE-EVENTS-00045.txt
\s*
^Admission[ ]Date: (.*?)	Discharge[ ]Date: (.*?)
^Date[ ]of[ ]Birth: (.*?)	Sex: (.*?)
^Service: (.*?)
^HISTORY[ ]OF[ ]PRESENT[ ]ILLNESS: (.*?)
^PAST[ ]MEDICAL[ ]HISTORY: (.*?)
^MEDICATIONS: (.*?)
^SUMMARY[ ]OF[ ]HOSPITAL[ ]COURSE: (.*?)
^DISCHARGE[ ]CONDITION: (.*?)
^DISCHARGE[ ]DISPOSITION: (.*?)
^DISCHARGE[ ]MEDICATIONS: (.*?)
^OTHER[ ]INSTRUCTIONS: (.*?)
^(\s* (?:Dr[.]\s?)? [[].*?)
^\(End[ ]of[ ]Report\)
""",

r"""	# modeled from NOTE-EVENTS-00064.txt
\s*
^Admission[ ]Date: (.*?)	Discharge[ ]Date: (.*?)
^Date[ ]of[ ]Birth: (.*?)	Sex: (.*?)
^Service: (.*?)
^CHIEF[ ]COMPLAINT: (.*?)
^HISTORY[ ]OF[ ]PRESENT[ ]ILLNESS: (.*?)
^PAST[ ]MEDICAL[ ]HISTORY: (.*?)
^ALLERGIES: (.*?)
^MEDICATIONS[ ]ON[ ]ADMISSION: (.*?)
^SOCIAL[ ]HISTORY: (.*?)
^INITIAL[ ]PHYSICAL[ ]EXAMINATION: (.*?)
^HOSPITAL[ ]COURSE[ ]BY[ ]SYSTEMS: (.*?)
^DISCHARGE[ ]CONDITION: (.*?)
^DISCHARGE[ ]PLACEMENT: (.*?)
^DISCHARGE[ ]DIAGNOSES: (.*?)
^DISCHARGE[ ]MEDICATIONS: (.*?)
^FOLLOWUP: (.*?)
^(\s* (?:Dr[.]\s?)? [[].*?)
^\(End[ ]of[ ]Report\)
""",

r"""	# modeled from NOTE-EVENTS-00458.txt
\s*
^Admission[ ]Date: (.*?)	Discharge[ ]Date: (.*?)
^Date[ ]of[ ]Birth: (.*?)	Sex: (.*?)
^Service: (.*?)
^HISTORY[ ]OF[ ]PRESENT[ ]ILLNESS: (.*?)
^PAST[ ]MEDICAL[ ]HISTORY: (.*?)
^MEDICATIONS[ ]ON[ ]ADMISSION: (.*?)
^ALLERGIES: (.*?)
^SOCIAL[ ]HISTORY: (.*?)
^FAMILY[ ]HISTORY: (.*?)
^PHYSICAL[ ]EXAMINATION: (.*?)
^LABORATORY[ ]DATA: (.*?)
^HOSPITAL[ ]COURSE: (.*?)
^DISCHARGE[ ]DIAGNOSIS: (.*?)
^DISCHARGE[ ]STATUS: (.*?)
^DISCHARGE[ ]DISPOSITION: (.*?)
^DISCHARGE[ ]MEDICATIONS: (.*?)
^(\s* (?:Dr[.]\s?)? [[].*?)
^\(End[ ]of[ ]Report\)
""",

r"""	# modeled from NOTE-EVENTS-00498.txt
\s*
^Admission[ ]Date: (.*?)	Discharge[ ]Date: (.*?)
^Date[ ]of[ ]Birth: (.*?)	Sex: (.*?)
^Service: (.*?)
^HISTORY[ ]OF[ ]THE[ ]PRESENT[ ]ILLNESS: (.*?)
^PAST[ ]MEDICAL[ ]HISTORY: (.*?)
^ADMISSION[ ]MEDICATIONS: (.*?)
^ALLERGIES: (.*?)
^SOCIAL[ ]HISTORY: (.*?)
^HOSPITAL[ ]COURSE: (.*?)
^CONDITION[ ]AT[ ]DISCHARGE: (.*?)
^STATUS[ ]AT[ ]DISCHARGE: (.*?)
^DISCHARGE[ ]MEDICATIONS: (.*?)
^DISCHARGE[ ]INSTRUCTIONS: (.*?)
^FOLLOW\-UP: (.*?)
^(\s* (?:Dr[.]\s?)? [[].*?)
^\(End[ ]of[ ]Report\)
""",

r"""	# modeled from NOTE-EVENTS-01050.txt
\s*
^Admission[ ]Date: (.*?)	Discharge[ ]Date: (.*?)
^Date[ ]of[ ]Birth: (.*?)	Sex: (.*?)
^Service: (.*?)
^Allergies: (.*?)
^Attending: (.*?)
^Major[ ]Surgical[ ]or[ ]Invasive[ ]Procedure: (.*?)
^History[ ]of[ ]Present[ ]Illness: (.*?)
^Past[ ]Medical[ ]History: (.*?)
^Social[ ]History: (.*?)
^Physical[ ]Exam: (.*?)
^Pertinent[ ]Results: (.*?)
^Brief[ ]Hospital[ ]Course: (.*?)
^Medications[ ]on[ ]Admission: (.*?)
^Discharge[ ]Medications: (.*?)
^Discharge[ ]Disposition: (.*?)
^Discharge[ ]Diagnosis: (.*?)
^Discharge[ ]Condition: (.*?)
^Discharge[ ]Instructions: (.*?)
^Followup[ ]Instructions: (.*?)
^(\s* (?:Dr[.]\s?)? [[].*?)
^\(End[ ]of[ ]Report\)
""",

r"""	# modeled from NOTE-EVENTS-00511.txt
\s*
^Admission[ ]Date: (.*?)	Discharge[ ]Date: (.*?)
^Date[ ]of[ ]Birth: (.*?)	Sex: (.*?)
^Service: (.*?)
^Allergies: (.*?)
^Attending: (.*?)
^Chief[ ]Complaint: (.*?)
^Major[ ]Surgical[ ]or[ ]Invasive[ ]Procedure: (.*?)
^History[ ]of[ ]Present[ ]Illness: (.*?)
^Past[ ]Medical[ ]History: (.*?)
^Social[ ]History: (.*?)
^Family[ ]History: (.*?)
^Physical[ ]Exam: (.*?)
^Pertinent[ ]Results: (.*?)
^Brief[ ]Hospital[ ]Course: (.*?)
^Medications[ ]on[ ]Admission: (.*?)
^Discharge[ ]Medications: (.*?)
^Discharge[ ]Disposition: (.*?)
^Facility: (.*?)
^Discharge[ ]Diagnosis: (.*?)
^Discharge[ ]Condition: (.*?)
^Discharge[ ]Instructions: (.*?)
^(\s* (?:Dr[.]\s?)? [[].*?)
^\(End[ ]of[ ]Report\)
""",

r"""	# modeled from NOTE-EVENTS-01010.txt
\s*
^Admission[ ]Date: (.*?)	Discharge[ ]Date: (.*?)
^Date[ ]of[ ]Birth: (.*?)	Sex: (.*?)
^Service: (.*?)
^Allergies: (.*?)
^Attending: (.*?)
^Chief[ ]Complaint: (.*?)
^Major[ ]Surgical[ ]or[ ]Invasive[ ]Procedure: (.*?)
^History[ ]of[ ]Present[ ]Illness: (.*?)
^Past[ ]Medical[ ]History: (.*?)
^Social[ ]History: (.*?)
^Family[ ]History: (.*?)
^Physical[ ]Exam: (.*?)
^Pertinent[ ]Results: (.*?)
^Brief[ ]Hospital[ ]Course: (.*?)
^Medications[ ]on[ ]Admission: (.*?)
^Discharge[ ]Disposition: (.*?)
^Discharge[ ]Diagnosis: (.*?)
^Discharge[ ]Condition: (.*?)
^Discharge[ ]Instructions: (.*?)
^Followup[ ]Instructions: (.*?)
^(\s* (?:Dr[.]\s?)? [[].*?)
^\(End[ ]of[ ]Report\)
""",

r"""	# modeled from NOTE-EVENTS-0579.txt
\s*
^Admission[ ]Date: (.*?)	Discharge[ ]Date: (.*?)
^Date[ ]of[ ]Birth: (.*?)	Sex: (.*?)
^HISTORY[ ]OF[ ]PRESENT[ ]ILLNESS: (.*?)
^PRENATAL[ ]SCREENS: (.*?)
^PHYSICAL[ ]EXAMINATION[ ]ON[ ]PRESENTATION: (.*?)
^HOSPITAL[ ]COURSE[ ]BY[ ]SYSTEM: (.*?)
^CONDITION[ ]AT[ ]DISCHARGE: (.*?)
^DISCHARGE[ ]DISPOSITION: (.*?)
^PEDIATRICIAN: (.*?)
^CARE[ ]RECOMMENDATIONS: (.*?)
^IMMUNIZATIONS[ ]RECEIVED: (.*?)
^IMMUNIZATIONS[ ]RECOMMENDED: (.*?)
^FOLLOWUP[ ]APPOINTMENTS: (.*?)
^DISCHARGE[ ]DIAGNOSES: (.*?)
^(\s* (?:Dr[.]\s?)? [[].*?)
^\(End[ ]of[ ]Report\)
""",


]

def readnote(path):
    with open(path) as f:
        nheads = f.readline().strip().split("_:-:_")
        heads = nheads[:-1]
        regex = "^" + "_:-:_".join("(?P<%s>.*?)" % (h) for h in heads) + "$"
        matches = re.finditer(regex, f.read(), re.MULTILINE | re.DOTALL)
    notes = map(lambda m: m.groupdict(), matches)
    return nheads, notes

def matchtemplates(note, temps=TEMPLATES):
	return [i for i,t in enumerate(temps) if re.match(t, note['TEXT'], REGEX_FLAGS)]
	
def matchnotes(doc, path=NURSE_NOTES_PATH):
	nheads, notes = readnote(os.path.join(path, doc))
	return map(matchtemplates, notes)

def matchall(limit=None):
	m = {}
	for i, doc in enumerate(os.listdir(NURSE_NOTES_PATH)[:limit]):
		if i % 10 == 0:
			print "%d..\r" % (i)
		m[doc] = [item for sublist in matchnotes(doc) for item in sublist]
	return m
    #return dict((doc, matchnotes(doc)) for doc in os.listdir(NURSE_NOTES_PATH)[:limit])

m = matchall()
print "------------------"
a = set()
b = set()   
for e in m:
	(a if m[e] else b).add(e)

if a:
	print "Example match capture"
	r = list(a)[0]
	print r
	nheads, notes = readnote(os.path.join(NURSE_NOTES_PATH, r))
	note = notes[0]['TEXT']
	for i, g in enumerate(re.match(TEMPLATES[m[r][0]], note, REGEX_FLAGS).groups()):
		print "<%d>\n%s\n</%d>" % (i, g, i)
		
if b:
	print "Some non-matches:"
	print list(b)[:10]
	print

print "Match vs Non:", len(a), len(b), "using", len(TEMPLATES), "templates"
print
	
def magic(s):
	s = re.split('\n\s*\n', s)
	for e in s:
		e = e.split('\n')
		print 'r"""	# modeled from NOTE-EVENTS-%s.txt' % (e[0])
		print '\s*'
		print '^Admission[ ]Date: (.*?)	Discharge[ ]Date: (.*?)'
		print '^Date[ ]of[ ]Birth: (.*?)	Sex: (.*?)'
		print '^Service: (.*?)'
		for i in e[1:]:
			if i:
				print "^%s: (.*?)" % (re.escape(i).replace('\ ', '[ ]'))
		print '^(\s* (?:Dr[.]\s?)? [[].*?)'
		print '^\(End[ ]of[ ]Report\)'
		print '""",'
		print

