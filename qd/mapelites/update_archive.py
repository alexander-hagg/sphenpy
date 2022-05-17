def update_archive(replaced, replacement, archive, fitness, genes, features):
    # Update fitness values
    a_fitness = archive['fitness']
    for f in range(len(replaced)):
        a_fitness[replaced[f][0],replaced[f][1]] = fitness[replacement[f]]
    archive.update({'fitness': a_fitness})

    # Update feature values
    a_features = archive['features']
    for f in range(len(replaced)):
        a_features[replaced[f][0],replaced[f][1], :] = features[replacement[f], :]
    archive.update({'features': a_features})

    # Update gene values
    a_genes = archive['genes']
    for f in range(len(replaced)):
        a_genes[replaced[f][0],replaced[f][1],:] = genes[replacement[f],:]
    archive.update({'genes': a_genes})
    
    return archive
