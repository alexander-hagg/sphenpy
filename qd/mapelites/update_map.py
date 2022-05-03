def update_map(replaced, replacement, archive, fitness, genes, features):
    # Update fitness values
    a_fitness = archive.get('fitness')
    for f in range(len(replaced)):
        a_fitness[replaced[f][0],replaced[f][1]] = fitness[:,replacement[f]]
    archive.update({'fitness': a_fitness})
    
    # Update feature values
    a_features = archive.get('features')
    for f in range(len(replaced)):
        a_features[:,replaced[f][0],replaced[f][1]] = features[:,replacement[f]]
    archive.update({'features': a_features})
    
    # Update gene values
    a_genes = archive.get('genes')
    for f in range(len(replaced)):
        a_genes[replaced[f][0],replaced[f][1],:] = genes[replacement[f],:]
    archive.update({'genes': a_genes})
    
    return archive
