import csv
import shutil
import os


def read_csv(filename):
    classid_list = []
    path_list = []

    with open(filename, 'r') as file:
        csv_reader = csv.DictReader(file)
        headers = csv_reader.fieldnames
        if "ClassId" not in headers or "Path" not in headers:
            print("Die CSV-Datei enthält nicht die erforderlichen Spalten.")
            return classid_list, path_list

        for row in csv_reader:
            classid_list.append(row['ClassId'])
            path_list.append(row['Path'])

    return classid_list, path_list


def save_image(image_path, folder_path, i):
    try:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print("Der Zielordner wurde erfolgreich erstellt.")

        shutil.copy(image_path, folder_path)
        print(str(i) + ". Bild erfolgreich gespeichert.")
    except FileNotFoundError:
        print("Die angegebene Bilddatei wurde nicht gefunden.")
    except IsADirectoryError:
        print("Der angegebene Zielordner konnte nicht erstellt werden.")
    except shutil.SameFileError:
        print("Das Bild existiert bereits im Zielordner.")


if __name__ == "__main__":
    filename = './dataset/Test_label.csv'
    classid_list, path_list = read_csv(filename)

    print('Konvertiere ' + str(len(classid_list)) + " Bilder...")
    for i in range(0, len(classid_list)):
        image_path = 'dataset/Test/00/' + path_list[i][5:]
        folder_path = 'Validation/' + str(classid_list[i])
        save_image(image_path, folder_path, i)



